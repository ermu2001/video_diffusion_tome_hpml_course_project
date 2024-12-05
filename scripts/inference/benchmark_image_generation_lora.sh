
# # For example:
# lora_weight_path="MODELS/train_midjv6_bsz128_step1000_lr1e-5/lcm-sd35-distilled/transformer_lora"
# lora_weight_path="MODELS/train_cc12m_step1000_bsz128_lr5e-5/lcm-sd35-distilled/transformer_lora"
# requires two arguments:
# 1. lora_weight_path: the path to the lora weights
# 2. output_name: the name of the output directory
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 lora_weight_path output_name"
    exit 1
fi

lora_weight_path=${1}
output_name=${2}
prompt_sources=${3:-"coco"}
num_steps_list=(2 4 8 16 32 64)

# benchmarking with short scheduler (guidance scale=0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate get_pipeline=get_sd35_lora_pipeline get_pipeline.lora_weight_path=${lora_weight_path} \
        generate_kwargs=short \
        generate_kwargs.num_inference_steps=$num_steps \
        output_root_dir=RESULTS/${output_name}_no_cfg \
        prompt_sources=${prompt_sources}
done

# benchmarking with origin scheduler (guidance scale=7.0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate get_pipeline=get_sd35_lora_pipeline get_pipeline.lora_weight_path=${lora_weight_path} \
        generate_kwargs.num_inference_steps=$num_steps \
        output_root_dir=RESULTS/${output_name}_cfg \
        prompt_sources=${prompt_sources}
done

