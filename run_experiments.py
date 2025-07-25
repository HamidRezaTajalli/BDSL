from pathlib import Path
import subprocess


job_executer_files_path = Path("./job_executer_files")
if not job_executer_files_path.exists():
    job_executer_files_path.mkdir(parents=True, exist_ok=True)


template_file_address = Path("./job_executer.sh")

# Parameter lists based on utils.py arguments
model_list = ["resnet18", "resnet50", "vgg11", "vgg19", "densenet121", "vit_b16"]
dataset_list = ["CIFAR10"]
num_clients_list = [10]
num_rounds_list = [40]
epochs_per_client_list = [1]
batch_size_list = [32]
cut_layer_list = [1]
poisoning_rate_list = [0.9]
target_label_list = [0]

# Experiment configuration
num_exp = 1
exp_num_list = range(0, num_exp)
checkpoint_dir = "./split_learning_checkpoints"

######### Split Learning Backdoor Experiments #############################
###############################################################

for exp_num in exp_num_list:
    for dataset in dataset_list:
        for model in model_list:
            for num_clients in num_clients_list:
                for num_rounds in num_rounds_list:
                    for epochs_per_client in epochs_per_client_list:
                        for cut_layer in cut_layer_list:
                            for poisoning_rate in poisoning_rate_list:
                                for target_label in target_label_list:
                                    job_script_file = f"exp_{exp_num}_{dataset}_{model}_{num_clients}_{num_rounds}_{epochs_per_client}_{cut_layer}_{poisoning_rate}_{target_label}.sh"
                                    job_script_file_address = job_executer_files_path / Path(job_script_file)

                                    # Read the template and append the command to run the experiment
                                    with open(template_file_address, 'r') as template_file:
                                        template_content = template_file.read()
                                    
                                    # write the content to the job script file
                                    with open(job_script_file_address, 'w') as job_script_file:
                                        job_script_file.write(template_content)

                                    # create the command to run the experiment
                                    command = f"srun python utils.py --model {model} --dataset {dataset} --num_clients {num_clients} --num_rounds {num_rounds} --epochs_per_client {epochs_per_client} --batch_size {batch_size_list[0]} --cut_layer {cut_layer} --checkpoint_dir {checkpoint_dir} --poisoning_rate {poisoning_rate} --target_label {target_label} --exp_num {exp_num}"

                                    # append the command to the job script file
                                    with open(job_script_file_address, 'a') as job_script_file:
                                        job_script_file.write("\n")  # Ensure there's a newline before adding the command
                                        job_script_file.write(command)
                                        
                                    # Make the script executable
                                    subprocess.run(['chmod', '+x', str(job_script_file_address)])
                                    # Submit the job script to SLURM
                                    subprocess.run(['sbatch', str(job_script_file_address)])

#######################################################################

####################### Reduced Parameter Set Experiments ###############
#######################################################################







