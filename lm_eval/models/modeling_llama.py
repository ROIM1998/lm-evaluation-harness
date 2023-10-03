import transformers

from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaConfig,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
)

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Union, Optional, Tuple, List
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)

class ElasticLlamaRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps)
        self.retained_indices = None

    def forward(self, hidden_states, hidden_z=None, use_teacher=False):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if self.retained_indices is not None and not use_teacher:
            if hidden_states.shape[-1] != self.retained_indices.shape[-1]:
                variance = hidden_states[..., self.retained_indices].pow(2).mean(-1, keepdim=True)
            else:
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
            weight_use = self.weight.index_select(0, self.retained_indices)
        elif hidden_z is not None:
            remaining_indices = torch.where(~hidden_z.eq(0))[0]
            variance = hidden_states[..., remaining_indices].pow(2).mean(-1, keepdim=True)
            weight_use = self.weight
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            weight_use = self.weight
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight_use * hidden_states.to(input_dtype)
    

class ElasticLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = ElasticLlamaModel(config)
        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_teacher: bool =False,
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        pass_mask: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Using bottom-up pruning, disable layer-level zs
        head_layer_z = None
        mlp_z = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_teacher=use_teacher,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
class ElasticLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([ElasticLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = ElasticLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Re-initialize weights and apply final processing
        self.post_init()
        self.retained_indices = None
        
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.layers[layer].self_attn.prune_heads(heads)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_teacher: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # TODO: support mask shape conversion, and retained_indices use
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        
        if self.retained_indices is not None and not use_teacher:
            hidden_states = hidden_states.index_select(-1, self.retained_indices)

        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    use_teacher=use_teacher,
                    head_z=head_z[idx] if head_z is not None else None,
                    head_layer_z=head_layer_z[idx] if head_layer_z is not None else None,
                    intermediate_z=intermediate_z[idx] if intermediate_z is not None else None,
                    mlp_z=mlp_z[idx] if mlp_z is not None else None,
                    hidden_z=hidden_z,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, hidden_z, use_teacher=use_teacher)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        
class ElasticLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = ElasticLlamaAttention(config)
        self.mlp = ElasticLlamaMLP(config)
        self.input_layernorm = ElasticLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ElasticLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        use_teacher: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states, hidden_z, use_teacher=use_teacher)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            use_teacher=use_teacher,
            head_z=head_z,
            head_layer_z=head_layer_z,
        )
        if hidden_states is None:
            hidden_states = residual
        else:
            hidden_states = residual + hidden_states
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, hidden_z, use_teacher=use_teacher)
        hidden_states = self.mlp(
            hidden_states,
            use_teacher=use_teacher,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
        )
        if hidden_states is None:
            hidden_states = residual
        else:
            hidden_states = residual + hidden_states
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
        
        
class ElasticLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.num_teacher_heads = self.num_heads
        self.num_teacher_key_value_heads = self.num_key_value_heads
        self.teacher_hidden_size = self.hidden_size
        self.pruned_heads = set()
        self.teacher_pruned_heads = set()
        self.head_size = config.hidden_size // config.num_attention_heads
        
    def project(self, hidden_states, proj_layer: nn.Linear, use_teacher: bool = False):
        hidden_states = proj_layer(hidden_states)
        return hidden_states
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_teacher:bool = None,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        num_heads_use = self.num_teacher_heads if use_teacher else self.num_heads
        num_key_value_heads_use = self.num_teacher_key_value_heads if use_teacher else self.num_key_value_heads
        hidden_size_use = self.teacher_hidden_size if use_teacher else self.hidden_size

        if self.v_proj is None or num_heads_use == 0:
            return None, None, None

        if self.config.pretraining_tp > 1:
            key_value_slicing = (num_key_value_heads_use * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (num_heads_use * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.project(hidden_states, self.q_proj, use_teacher=use_teacher)
            key_states = self.project(hidden_states, self.k_proj, use_teacher=use_teacher)
            value_states = self.project(hidden_states, self.v_proj, use_teacher=use_teacher)

        query_states = query_states.view(bsz, q_len, num_heads_use, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads_use, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads_use, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, num_heads_use, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads_use, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # shape (bsz, num_heads, q_len, head_dim)
        
        if head_z is not None:
            head_z = head_z.view(-1, self.num_heads).unsqueeze(-1).unsqueeze(-1) # (1, num_heads, 1, 1)
            attn_output = attn_output * head_z

        if attn_output.size() != (bsz, num_heads_use, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads_use, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size_use)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(hidden_size_use // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(hidden_size_use // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class ElasticLlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        
    def forward(self, x, use_teacher=False, intermediate_z=None, mlp_z=None):
        if self.up_proj is None or (self.block_retained_indices is not None and self.block_retained_indices.numel() == 0 and not use_teacher):
            return None
        
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj)
            if intermediate_z is not None:
                intermediate_states = intermediate_states.mul(intermediate_z)
            intermediate_states = intermediate_states.split(slice, dim=2)
            
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            gated_x = self.gate_proj(x)
            gated_x = self.act_fn(gated_x)
            
            upped_x = self.up_proj(x)
            
            upped_x = gated_x * upped_x
            if intermediate_z is not None:
                upped_x = upped_x.mul(intermediate_z)
                
            down_proj = self.down_proj(upped_x)

        return down_proj
        