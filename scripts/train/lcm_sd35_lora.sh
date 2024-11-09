export PYTHONPATH=.:${PYTHONPATH}
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
accelerate launch accediff/train/lcm_sd35_lora.py