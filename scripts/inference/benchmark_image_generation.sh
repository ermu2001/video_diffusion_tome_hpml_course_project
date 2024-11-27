
benchmark_origin () {
    echo "Benchmarking image generation with text-to-image models"
    python -m accediff.inference.generate --config-name config_text_to_image
}


benchmark_origin