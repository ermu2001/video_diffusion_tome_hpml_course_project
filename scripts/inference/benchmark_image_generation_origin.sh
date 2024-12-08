
num_steps_list=(2 4 8 16 32 64)
output_name=origin_baseline
# prompt_sources="midjv6"
prompt_sources="coco"
repo_id="stabilityai/stable-diffusion-3.5-medium"

# benchmarking with short scheduler (guidance scale=0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate \
        generate_kwargs=short \
        generate_kwargs.num_inference_steps=$num_steps \
        output_root_dir=RESULTS/${prompt_sources}_${output_name}_no_cfg \
        prompt_sources=${prompt_sources} \
        get_pipeline.repo_id=${repo_id}
done

# benchmarking with origin scheduler (guidance scale=7.0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate \
        generate_kwargs.num_inference_steps=$num_steps \
        output_root_dir=RESULTS/${prompt_sources}_${output_name}_cfg \
        prompt_sources=${prompt_sources} \
        get_pipeline.repo_id=${repo_id}
done



bash scripts/evaluation/evaluate_image_clip_score.sh RESULTS/${prompt_sources}_${output_name}_cfg
bash scripts/evaluation/evaluate_image_clip_score.sh RESULTS/${prompt_sources}_${output_name}_no_cfg
bash scripts/evaluation/evaluate_image_fid_score.sh RESULTS/${prompt_sources}_${output_name}_cfg ${prompt_sources}
bash scripts/evaluation/evaluate_image_fid_score.sh RESULTS/${prompt_sources}_${output_name}_no_cfg ${prompt_sources}
