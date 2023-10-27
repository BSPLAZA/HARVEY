import os
import subprocess

def run_in_new_cmd(command):
    cmd_command = f'cmd.exe /k "{command}"'
    subprocess.Popen(cmd_command, shell=True)

# Commands to be executed
commands = [
    'python -c "import torch; print('Device under use: \033[92m' + (torch.cuda.get_device_name() + '\033[0m (GPU)' if torch.cuda.is_available() else '\033[94m' + __import__('platform').processor() + '\033[0m (CPU)'))"'
]

# Join the commands using '&&' to run them sequentially in the same cmd session
joined_commands = ' && '.join(commands)

run_in_new_cmd(joined_commands)
    

