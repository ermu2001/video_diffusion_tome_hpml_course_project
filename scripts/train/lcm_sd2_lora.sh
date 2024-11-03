export PYTHONPATH=.:${PYTHONPATH}
accelerate launch accediff/train/lcm_sd2_lora.py
