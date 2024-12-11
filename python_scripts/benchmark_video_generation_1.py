import itertools
import math
import os
# benchmark_origin () {
#     echo "Benchmarking video generation with text-to-video models"
#     python -m accediff.inference.generate_video_text_to_video 
# }

# benchmark_naive_tome_3d () {
#     echo "Benchmarking video generation with text-to-video models"
#     python -m accediff.inference.generate_video_text_to_video \
#         get_pipeline=get_covvideox_pipeline_tome_3d
# }



# benchmark_naive_tome_attnbin () {
#     echo "Benchmarking video generation with text-to-video models"
#     python -m accediff.inference.generate_video_text_to_video \
#         get_pipeline=get_covvideox_pipeline_tome_attnbin
# }

def benchmark_origin():
    print("Benchmarking video generation with text-to-video models")
    os.system("python -m accediff.inference.generate_video_text_to_video")

def benchmark_naive_tome_3d(
        merge_ratio,
        st,
        sx,
        sy,
):
    print("Benchmarking video generation with text-to-video models")
    hyper_params = f"""\
        get_pipeline.hook_kwargs.merge_ratio={merge_ratio} \
        get_pipeline.hook_kwargs.st={st} \
        get_pipeline.hook_kwargs.sx={sx} \
        get_pipeline.hook_kwargs.sy={sy} \
    """
    cmd = f"python -m accediff.inference.generate_video_text_to_video get_pipeline=get_covvideox_pipeline_tome_3d {hyper_params}"
    print(cmd)
    os.system(cmd)



def benchmark_naive_tome_attnbin(
    merge_ratio,
    sb,
):
    print("Benchmarking video generation with text-to-video models")
    hyper_params = f"""\
        get_pipeline=get_covvideox_pipeline_tome_attnbin \
        get_pipeline.hook_kwargs.merge_ratio={merge_ratio} \
        get_pipeline.hook_kwargs.sb={sb} \
    """
    cmd = f"python -m accediff.inference.generate_video_text_to_video get_pipeline=get_covvideox_pipeline_tome_3d {hyper_params}"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    merge_ratios = [0.3, 0.5]
    st_sx_sxs = [
        (1, 3, 3),
        (2, 3, 3),
        (4, 6, 9),
    ]

    benchmark_origin()


    # 3d tome
    for merge_ratio, (st, sx, sy) in itertools.product(merge_ratios, st_sx_sxs):
        benchmark_naive_tome_3d(merge_ratio, st, sx, sy)

    # attnbin tome
    for merge_ratio, (st, sx, sy) in itertools.product(merge_ratios, st_sx_sxs):
        sb = math.prod((st, sx, sy))
        benchmark_naive_tome_attnbin(merge_ratio, sb)