from pathlib import Path
import subprocess


job_executer_files_path = Path("./job_executer_files")
if not job_executer_files_path.exists():
    job_executer_files_path.mkdir(parents=True, exist_ok=True)


template_file_address = Path("./job_executer.sh")

# Parameter lists based on utils.py arguments
model_list = ["resnet18", "resnet50", "vgg19", "densenet121", "vit_b16"]
model_list = ["resnet18"]
dataset_list = ["CIFAR10"]
num_clients_list = [10]
num_rounds_list = [60]
epochs_per_client_list = [1]
batch_size_list = [128]
cut_layer_list = [1]
poisoning_rate_list = [0.2]
target_label_list = [0]
num_workers = 14

# Additional parameter lists for attack_only.py
attack_list = ["badnet", "blend", "sig"]
attack_list = ["blend"]
trigger_size_list = [0.08]
attack_mode_list = ["all-to-one"]

# Experiment configuration
num_exp = 1
exp_num_list = range(0, num_exp)




###########################################################################
######################### attack only experiments #########################
###########################################################################

# for exp_num in exp_num_list:
#     for dataset in dataset_list:
#         for model in model_list:
#             for num_clients in num_clients_list:
#                 for num_rounds in num_rounds_list:
#                     for epochs_per_client in epochs_per_client_list:
#                         for cut_layer in cut_layer_list:
#                             for poisoning_rate in poisoning_rate_list:
#                                 for target_label in target_label_list:
#                                     for attack in attack_list:
#                                         for trigger_size in trigger_size_list:
#                                             for attack_mode in attack_mode_list:
#                                                 job_script_file = f"attack_exp_{exp_num}_{dataset}_{model}_{num_clients}_{num_rounds}_{epochs_per_client}_{cut_layer}_{poisoning_rate}_{target_label}_{attack}_{trigger_size}_{attack_mode}.sh"
#                                                 job_script_file_address = job_executer_files_path / Path(job_script_file)

#                                                 # Read the template and append the command to run the experiment
#                                                 with open(template_file_address, 'r') as template_file:
#                                                     template_content = template_file.read()
                                                
#                                                 # write the content to the job script file
#                                                 with open(job_script_file_address, 'w') as job_script_file:
#                                                     job_script_file.write(template_content)

#                                                 # create the command to run the attack_only experiment
#                                                 command = f"srun python attack_only.py --model {model} --dataset {dataset} --num_clients {num_clients} --num_rounds {num_rounds} --epochs_per_client {epochs_per_client} --batch_size {batch_size_list[0]} --cut_layer {cut_layer} --poisoning_rate {poisoning_rate} --target_label {target_label} --exp_num {exp_num} --attack {attack} --trigger_size {trigger_size} --attack_mode {attack_mode} --num_workers {num_workers}"

#                                                 # append the command to the job script file
#                                                 with open(job_script_file_address, 'a') as job_script_file:
#                                                     job_script_file.write("\n")  # Ensure there's a newline before adding the command
#                                                     job_script_file.write(command)
                                                    
#                                                 # Make the script executable
#                                                 subprocess.run(['chmod', '+x', str(job_script_file_address)])
#                                                 # Submit the job script to SLURM
#                                                 subprocess.run(['sbatch', str(job_script_file_address)])


####################################################################################





############################## step by step experiments ##############################
####################################################################################

for exp_num in exp_num_list:
    for dataset in dataset_list:
        for model in model_list:
            for num_clients in num_clients_list:
                for num_rounds in num_rounds_list:
                    for epochs_per_client in epochs_per_client_list:
                        for cut_layer in cut_layer_list:
                            for poisoning_rate in poisoning_rate_list:
                                for target_label in target_label_list:
                                    for attack in attack_list:
                                        for trigger_size in trigger_size_list:
                                            for attack_mode in attack_mode_list:
                                                job_script_file = f"step_by_step_exp_{exp_num}_{dataset}_{model}_{num_clients}_{num_rounds}_{epochs_per_client}_{cut_layer}_{poisoning_rate}_{target_label}_{attack}_{trigger_size}_{attack_mode}.sh"
                                                job_script_file_address = job_executer_files_path / Path(job_script_file)

                                                # Read the template and append the command to run the experiment
                                                with open(template_file_address, 'r') as template_file:
                                                    template_content = template_file.read()
                                                
                                                # write the content to the job script file
                                                with open(job_script_file_address, 'w') as job_script_file:
                                                    job_script_file.write(template_content)

                                                # create the command to run the step_by_step experiment
                                                command = f"srun python step_by_step.py --model {model} --dataset {dataset} --num_clients {num_clients} --num_rounds {num_rounds} --epochs_per_client {epochs_per_client} --batch_size {batch_size_list[0]} --cut_layer {cut_layer} --poisoning_rate {poisoning_rate} --target_label {target_label} --exp_num {exp_num} --attack {attack} --trigger_size {trigger_size} --attack_mode {attack_mode} --num_workers {num_workers} --backbone_freeze_rounds 0 --backbone_freeze_prob 0.34"

                                                # append the command to the job script file
                                                with open(job_script_file_address, 'a') as job_script_file:
                                                    job_script_file.write("\n")  # Ensure there's a newline before adding the command
                                                    job_script_file.write(command)
                                                    
                                                # Make the script executable
                                                subprocess.run(['chmod', '+x', str(job_script_file_address)])
                                                # Submit the job script to SLURM
                                                subprocess.run(['sbatch', str(job_script_file_address)])

