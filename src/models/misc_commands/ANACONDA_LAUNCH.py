import os
import subprocess

def run_in_new_cmd(command):
    cmd_command = f'cmd.exe /k "{command}"'
    subprocess.Popen(cmd_command, shell=True)

# Commands to be executed
commands = [
    'call C:\\ProgramData\\anaconda3\\Scripts\\activate.bat HarveyVenv',
    f'cd {os.path.join(os.environ["USERPROFILE"], "Documents", "capstone", "HARVEY", "src", "models")}'
]

# Join the commands using '&&' to run them sequentially in the same cmd session
joined_commands = ' && '.join(commands)

run_in_new_cmd(joined_commands)