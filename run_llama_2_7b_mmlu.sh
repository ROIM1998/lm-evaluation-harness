weight_path=$1

python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,weight_path=${weight_path}" \
    --num_fewshot 5 \
    --tasks hendrycksTest-* \
    --device cuda:0 \
    --output_path "${weight_path}/mmlu_results" \