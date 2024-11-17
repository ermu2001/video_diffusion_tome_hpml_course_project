
source /scratch/yz10381/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/yz10381/miniconda3/envs/diffusers
python_script=accediff/train/lcm_sd35_lora.py
echo "extra args to ${python_script}: $@"

export PYTHONPATH=.:${PYTHONPATH}
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=info

# # # If srun, get the main process IP and ports

export MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MAIN_PROCESS_PORT=29500
export NUM_MACHINES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
export NUM_PROCESSES=$(($NUM_MACHINES * $SLURM_GPUS_PER_NODE))
export MACHINE_RANK=$SLURM_PROCID

main_process_ip=${MAIN_PROCESS_IP:-""}
main_process_port=${MAIN_PROCESS_PORT:-29500}
num_machines=${NUM_MACHINES:-1}
num_processes=${NUM_PROCESSES:-1}
machine_rank=${MACHINE_RANK:-0}

# # # # # # # # # # # # # # # # # # # # # # # # # # #  


echo "main_process_ip: ${main_process_ip}"
echo "main_process_port: ${main_process_port}"
echo "num_machines: ${num_machines}"
echo "num_processes: ${num_processes}"
echo "machine_rank: ${machine_rank}"

accelerate launch \
    --main_process_ip ${main_process_ip} \
    --num_machines ${num_machines} \
    --num_processes ${num_processes} \
    --mixed_precision no \
    --dynamo_backend no \
    --gpu_ids all \
    ${python_script} \
    $@