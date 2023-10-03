weight_path=$1

# ARC
python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-13b-hf,weight_path=${weight_path}" \
    --num_fewshot 25 \
    --tasks arc_challenge  \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${weight_path}/arc_results.json" 

# TruthfulQA
python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-13b-hf,weight_path=${weight_path}" \
    --num_fewshot 0 \
    --tasks truthfulqa_mc \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${weight_path}/truthfulqa_mc_results.json" 

# HellaSwag
python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-13b-hf,weight_path=${weight_path}" \
    --num_fewshot 10 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${weight_path}/hellaswag_results.json"

# MMLU
python main.py \
    --model hf-causal \
    --model_args "pretrained=meta-llama/Llama-2-13b-hf,weight_path=${weight_path}" \
    --num_fewshot 5 \
    --tasks hendrycksTest-* \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${weight_path}/mmlu_results.json" \