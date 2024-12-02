
# # For example:
# lora_weight_path="MODELS/train_midjv6_bsz128_step1000_lr1e-5/lcm-sd35-distilled/transformer_lora"
# lora_weight_path="MODELS/train_cc12m_step1000_bsz128_lr5e-5/lcm-sd35-distilled/transformer_lora"
lora_weight_path=...

num_steps_list=(2 4 8 16 32 64)

# benchmarking with origin scheduler (guidance scale=7.0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate get_pipeline=get_sd35_lora_pipeline get_pipeline.lora_weight_dir=${lora_weight_path} \
        generate_kwargs.num_inference_steps=$num_steps
done


# benchmarking with short scheduler (guidance scale=0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate get_pipeline=get_sd35_lora_pipeline get_pipeline.lora_weight_dir=${lora_weight_path} \
        generate_kwargs=short \
        generate_kwargs.num_inference_steps=$num_steps
done