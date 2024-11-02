# ACCEDIFF

## INSTALL
```shell
conda create -n diffusers python=3.10
conda activate diffusers
conda install -y -c nvidia cuda-toolkit
pip install -r requirements.txt
pip install -r requirements.manual.txt
pip install flash-attn --no-build-isolation
```