#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_a100
##SBATCH --partition=gpu_h100
#SBATCH --time=0-02:00:00
#SBATCH --mem=4GB
#SBATCH --output=script_logging/slurm_%A.out
#SBATCH --mail-type=END,FAIL                     # send email when job ends or fails
#SBATCH --mail-user=hamidreza.tajalli@ru.nl      # email address



# Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0


# srun python normal_sl.py --model resnet18 --dataset CIFAR10 --cut_layer 1 --num_clients 10 --num_rounds 40




# srun torchrun --standalone --nproc_per_node=4 /home/htajalli/prjs0962/repos/BA_NODE/temp_test_torchrun.py