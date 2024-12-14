# VIDEO DIFFUSION TOME
This is the repo for High-Performance Machine Learning (HPML) with professor Dr. Zehra Sura and Dr. Parijat Dube. The project intend to extend the original Token Merging for image diffusion acceleration to 3d Video Diffusion Models. This repo contains all the code for inference, evaluation and analysis for the project.

The rest of the README describes our project and instruction steps for running the experiments for reference.

# Project Milestone

This project explores how efficiency techniques originally developed for 2D image diffusion models can be effectively transferred to the more complex 3D video generation setting. Unlike image models that focus solely on spatial coherence within a single frame, video models must also maintain consistency across time, adding significant complexity.

Our investigation centers on the ToMe method—a training-free token merging approach—to reduce attention layers’ token counts and improve throughput.

Key Objectives:

- **Adapting 2D Methods to 3D:**
Apply token merging techniques from image-based diffusion models to video diffusion tasks, accounting for the added temporal dimension.

- **Finding Effective Designs:**
Identify which image-derived efficiency strategies preserve both spatial detail and temporal coherence in video outputs, and under what conditions these methods offer meaningful computational improvements.

- **Evaluating Performance Trade-offs:**
Assess the balance between acceleration and output quality, examining how token merging and similar methods influence video generation quality, inference speed, and memory usage.

# Repo Structure
The codebase was built up on [hydra](https://github.com/facebookresearch/hydra) for structured experiments running and documenting. Aside from this, we use the [diffusers](https://github.com/huggingface/diffusers) implementation for SOTA video diffusion generative models. The rest of this section explains the core implementation of experiments.

## ToMe Hooks
The code for registering token merging hooks to the diffusion transformer could be found in [pipeline_factory.py](https://github.com/ermu2001/video_diffusion_tome_hpml_course_project/blob/video_main/accediff/utils/factory/pipeline_factory.py).

## Prompts
The prompts for experiments are in [here](https://github.com/ermu2001/video_diffusion_tome_hpml_course_project/blob/video_main/static/prompts/20241130_open_chatgpt4o_videos.txt). The prompts are obtained form ChatGPT4o, prompted to generate plain text description for videos. To save computation, we generated 50 prompts and conduct all the experiments on these 50 prompts for consistency.

## Benchmarking
We mainly consider the end to end video generation time for a prompt as our monitoring factor for efficiency. The code could be found [here](https://github.com/ermu2001/video_diffusion_tome_hpml_course_project/blob/video_main/accediff/utils/utils.py#L21C1-L43C22). The first three generation was skipped as warm up.

## Experiments
To run our experiments, on could directly run the python scripts for multiple setting automatically. Mainly, the scripts we used are described as following:
1. [benchmark_video_generation_2.py](https://github.com/ermu2001/video_diffusion_tome_hpml_course_project/blob/video_main/python_scripts/benchmark_video_generation_2.py): Generating with blocks running ToMe alternatively with multiple ToMe strategy, including different strides for 3d Token Merging and Attnbin merging.
2. [benchmark_video_generation_3.py](https://github.com/ermu2001/video_diffusion_tome_hpml_course_project/blob/video_main/python_scripts/benchmark_video_generation_3.py): Generating with all block running ToMe, remains are the same as above.

## Evaluation
For quality evaluation, we follow the [VBench](https://github.com/Vchitect/VBench), and run the customized benchmark with out video. The running for evaluation could be found here [TODO].

# INSTALL
To install the environment, we recommend running the following commands.

```shell
conda create -n diffusers python=3.10
conda activate diffusers
conda install -y -c nvidia cuda-toolkit
pip install -r requirements.txt
pip install -r requirements.manual.txt
```
# Example Running
To run the experiments, first install the environment as described in the previous section. As described in the [Structure](https://github.com/ermu2001/video_diffusion_tome_hpml_course_project/edit/main/README.md#experiments) section, one are encouraged to directly run these command for experiments:
```shell
conda activate diffusers
python python_scripts/benchmark_video_generation_2.py
python python_scripts/benchmark_video_generation_1.py
```
# Results and Observations  
This section mainly introduces some of our results and experiments. All the videos we have generated could be find [here](https://huggingface.co/ermu2001/RESULTS), all our experiments are run a one H100.

## Merging Strategy
From experiments, the attnbin approach is suboptimal. This indicates that naively sorting with norm of tokens does not represent the structure for transformer tokens. More is to be done!
![2400702f9040c3e5466bbeb7274e532](https://github.com/user-attachments/assets/47aabfd0-5735-4f6e-bc27-27128e00e67f)

## Influence of ToMe
- At the heart of ToMe, we decrease the number of tokens going into the attention layers, this will run multiple time as the transformer is the core of diffusion iterative generation process. As a result throughput increased.
- After ToMe background subject, several variant’s consistency & consistency & motion smoothness increased. After case study, we find that some generated video only have sligh artifacts of the content. The fact that benchmarking such artifact may not be the concentratation of the benchmark. 
- Throughput distribution: attnbin < 3D. Due to the sort operation. We believe this is mainly because we have sort step in the attnbin function.

![dc8670ba82f673f5fb5a50a40b6d122](https://github.com/user-attachments/assets/cff07ac8-3b00-47ef-bf92-82cfdb4e0151)

# 💫 Acknowledgement

- [ToMe](https://github.com/dbolya/tomesd): Amazing Training Free Efficient Extension for Stable Diffusion
- [hydra](https://github.com/facebookresearch/hydra): Configuration system for experiments of repo
- [diffusers](https://github.com/huggingface/diffusers): SOTA image/video generative DL model pakcage
