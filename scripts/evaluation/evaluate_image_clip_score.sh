evaluate_image_clip_score () {
    generation_output_dir=$1
    echo "Evaluating image generation with text-to-image models of output directory $1"
    python -m accediff.evaluation.evaluate_image_clip_score \
        generation_output_dir=$generation_output_dir
}

# For example
# output_dir="RESULTS/mj_lr1e-05"
output_dir=${1} 
for generation_output_dir in $output_dir/*; do
    if [ ! -f "$generation_output_dir/benchmark_image_generation/generation_info.json" ]; then
        echo "Skipping $generation_output_dir, as it does not contain benchmark_image_generation/generation_info.json"
        continue
    fi
    evaluate_image_clip_score $generation_output_dir/benchmark_image_generation
done