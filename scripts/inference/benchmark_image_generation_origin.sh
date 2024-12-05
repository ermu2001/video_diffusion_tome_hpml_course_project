
num_steps_list=(2 4 8 16 32 64)
output_name=origin_baseline
prompt_sources="coco"
repo_id="stabilityai/stable-diffusion-3.5-medium"

# benchmarking with short scheduler (guidance scale=0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate \
        generate_kwargs=short \
        generate_kwargs.num_inference_steps=$num_steps \
        output_root_dir=RESULTS/${output_name}_no_cfg \
        prompt_sources=${prompt_sources} \
        get_pipeline.repo_id=${repo_id}
done

# benchmarking with origin scheduler (guidance scale=7.0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate \
        generate_kwargs.num_inference_steps=$num_steps \
        output_root_dir=RESULTS/${output_name}_cfg \
        prompt_sources=${prompt_sources} \
        get_pipeline.repo_id=${repo_id}
done
