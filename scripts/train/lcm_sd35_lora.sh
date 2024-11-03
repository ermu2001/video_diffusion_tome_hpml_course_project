export PYTHONPATH=.:${PYTHONPATH}
accelerate launch accediff/train/lcm_sd35_lora.py
