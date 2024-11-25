
benchmark_origin () {
    echo "Benchmarking video generation with text-to-video models"
    python -m accediff.inference.generate_video_text_to_video --config-name config_text_to_video
}

benchmark_naive_tome () {
    echo "Benchmarking video generation with text-to-video models"
    python -m accediff.inference.generate_video_text_to_video --config-name config_text_to_video_tome
}


benchmark_naive_tome
benchmark_origin