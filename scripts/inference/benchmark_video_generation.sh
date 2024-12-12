set -e
benchmark_origin () {
    echo "Benchmarking video generation with text-to-video models"
    python -m accediff.inference.generate_video_text_to_video 
}

benchmark_naive_tome_3d () {
    echo "Benchmarking video generation with text-to-video models"
    python -m accediff.inference.generate_video_text_to_video \
        get_pipeline=get_covvideox_pipeline_tome_3d
}



benchmark_naive_tome_attnbin () {
    echo "Benchmarking video generation with text-to-video models"
    python -m accediff.inference.generate_video_text_to_video \
        get_pipeline=get_covvideox_pipeline_tome_attnbin
}


benchmark_origin
benchmark_naive_tome_3d
benchmark_naive_tome_attnbin