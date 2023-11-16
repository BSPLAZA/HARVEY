import re

def simplify_env_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    simplified_lines = []
    for line in lines:
        # This regex pattern will keep only the package name and major version
        simplified_line = re.sub(r'([a-zA-Z0-9_-]+=[0-9]+(\.[0-9]+)?).*', r'\1', line)
        simplified_lines.append(simplified_line)

    with open(output_file, 'w') as file:
        file.writelines(simplified_lines)

# Usage
simplify_env_file('../../environment.yml', '../../environment.yml')
