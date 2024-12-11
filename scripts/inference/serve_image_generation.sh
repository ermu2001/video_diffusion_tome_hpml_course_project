# lora_weight_path=${1}
# python -m accediff.inference.serve get_pipeline=get_sd35_lora_pipeline get_pipeline.lora_weight_path=${lora_weight_path} \
#         generate_kwargs=short

python -m accediff.inference.serve get_pipeline=get_sd35_pipeline \
        generate_kwargs=short