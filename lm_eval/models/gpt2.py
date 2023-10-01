import os
import torch
import transformers
import torch.nn as nn

from typing import Optional, Union, Dict
from lm_eval.base import BaseLM
from tqdm import tqdm

from transformers.modeling_utils import prune_linear_layer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        weight_path=None,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            # Initialize new model and tokenizer instances
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(self.device)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            # TODO: add loading local pruned weights here
            if weight_path:
                model = self.model
                model_dtype = self.model.dtype
                assert isinstance(model, transformers.LlamaForCausalLM)
                weights = [torch.load(os.path.join(weight_path, f), map_location='cpu') for f in tqdm(os.listdir(weight_path)) if f.endswith(".bin") and 'arg' not in f]
                weights = {k: v for w in weights for k, v in w.items()}
                lora_keys = [k for k in weights.keys() if 'lora' in k]
                print("Dropping lora keys: %s, %d keys in total." % (lora_keys, len(lora_keys)))
                weights = {k: v for k, v in weights.items() if 'lora' not in k and 'transform' not in k}
                # Deducing pruned dimensions of the LLaMA model
                config: transformers.LlamaConfig = model.config
                vocab_size, model_dim = weights['model.embed_tokens.weight'].shape
                if vocab_size > config.vocab_size:
                    DEFAULT_PAD_TOKEN = "[PAD]"
                    DEFAULT_EOS_TOKEN = "</s>"
                    DEFAULT_BOS_TOKEN = "<s>"
                    DEFAULT_UNK_TOKEN = "<unk>"
                    special_tokens_dict = dict()
                    if self.tokenizer.pad_token is None:
                        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
                    if self.tokenizer.eos_token is None:
                        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
                    if self.tokenizer.bos_token is None:
                        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
                    if self.tokenizer.unk_token is None:
                        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

                    smart_tokenizer_and_embedding_resize(
                        special_tokens_dict=special_tokens_dict,
                        tokenizer=self.tokenizer,
                        model=model,
                    )
                if model_dim == config.hidden_size:
                    remained_dim = None
                else:
                    remained_dim = torch.arange(model_dim)
                    model.model.embed_tokens.weight = torch.nn.parameter.Parameter(
                        model.model.embed_tokens.weight.index_select(1, remained_dim).detach().clone()
                    )
                    model.model.embed_tokens.embedding_dim = remained_dim.shape[0]
                    model.model.norm.weight = torch.nn.parameter.Parameter(
                        model.model.norm.weight.index_select(0, remained_dim).detach().clone()
                    )
                dim_per_head = config.hidden_size // config.num_attention_heads
                for i_layer in range(config.num_hidden_layers):
                    head_dim = weights['model.layers.%d.self_attn.q_proj.weight' % i_layer].shape[0]
                    remained_head_dim = torch.arange(head_dim) if head_dim != config.hidden_size else None
                    if remained_head_dim is not None:
                        print("Pruning head dim to %d in layer %d" % (remained_head_dim.shape[0], i_layer))
                        model.model.layers[i_layer].self_attn.q_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.q_proj, remained_head_dim, dim=0)
                        model.model.layers[i_layer].self_attn.k_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.k_proj, remained_head_dim, dim=0)
                        model.model.layers[i_layer].self_attn.v_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.v_proj, remained_head_dim, dim=0)
                        model.model.layers[i_layer].self_attn.o_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.o_proj, remained_head_dim, dim=1)
                        model.model.layers[i_layer].self_attn.num_heads = remained_head_dim.shape[0] // dim_per_head
                        model.model.layers[i_layer].self_attn.num_key_value_heads = remained_head_dim.shape[0] // dim_per_head
                        model.model.layers[i_layer].self_attn.hidden_size = remained_head_dim.shape[0]
                    
                    ffn_dim = weights['model.layers.%d.mlp.up_proj.weight' % i_layer].shape[0]
                    remained_ffn_dim = torch.arange(ffn_dim) if ffn_dim != config.intermediate_size else None
                    if remained_ffn_dim is not None:
                        print("Pruning ffn dim to %d in layer %d" % (remained_ffn_dim.shape[0], i_layer))
                        model.model.layers[i_layer].mlp.up_proj = prune_linear_layer(model.model.layers[i_layer].mlp.up_proj, remained_ffn_dim, dim=0)
                        model.model.layers[i_layer].mlp.gate_proj = prune_linear_layer(model.model.layers[i_layer].mlp.gate_proj, remained_ffn_dim, dim=0)
                        model.model.layers[i_layer].mlp.down_proj = prune_linear_layer(model.model.layers[i_layer].mlp.down_proj, remained_ffn_dim, dim=1)
                        
                    if remained_dim is not None:
                        print("Pruning hidden dim to %d in layer %d" % (remained_dim.shape[0], i_layer))
                        model.model.layers[i_layer].self_attn.q_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.q_proj, remained_dim, dim=1)
                        model.model.layers[i_layer].self_attn.k_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.k_proj, remained_dim, dim=1)
                        model.model.layers[i_layer].self_attn.v_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.v_proj, remained_dim, dim=1)
                        model.model.layers[i_layer].self_attn.o_proj = prune_linear_layer(model.model.layers[i_layer].self_attn.o_proj, remained_dim, dim=0)
                        model.model.layers[i_layer].mlp.up_proj = prune_linear_layer(model.model.layers[i_layer].mlp.up_proj, remained_dim, dim=1)
                        model.model.layers[i_layer].mlp.gate_proj = prune_linear_layer(model.model.layers[i_layer].mlp.gate_proj, remained_dim, dim=1)
                        model.model.layers[i_layer].mlp.down_proj = prune_linear_layer(model.model.layers[i_layer].mlp.down_proj, remained_dim, dim=0)
                        model.model.layers[i_layer].input_layernorm.weight = nn.Parameter(
                            model.model.layers[i_layer].input_layernorm.weight.index_select(0, remained_dim).detach().clone()
                        )
                        model.model.layers[i_layer].post_attention_layernorm.weight = nn.Parameter(
                            model.model.layers[i_layer].post_attention_layernorm.weight.index_select(0, remained_dim).detach().clone()
                        )
                
                model.load_state_dict(weights, strict=True)
                model = model.to(model_dtype)

        else:
            raise TypeError(
                "Parameter pretrained should be of type str or transformers.PreTrainedModel"
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM
