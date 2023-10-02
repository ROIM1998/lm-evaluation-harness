weight_path=$1

python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,weight_path=${weight_path}" \
    --num_fewshot 25 \
    --tasks arc_challenge  \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${weight_path}/arc_results.json" \