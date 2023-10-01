weight_path=$1

python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,weight_path=${weight_path}" \
    --num_fewshot 0 \
    --tasks truthfulqa_mc \
    --device cuda:0 \
    --output_path "${weight_path}/truthfulqa_mc_results.json" \