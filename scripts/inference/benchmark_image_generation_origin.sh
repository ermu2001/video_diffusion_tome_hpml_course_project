
num_steps_list=(2 4 8 16 32 64)

# benchmarking with origin scheduler (guidance scale=7.0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate \
        generate_kwargs.num_inference_steps=$num_steps
done


# benchmarking with short scheduler (guidance scale=0)
for num_steps in "${num_steps_list[@]}"; do
    python -m accediff.inference.generate \
        generate_kwargs=short \
        generate_kwargs.num_inference_steps=$num_steps \
        
done