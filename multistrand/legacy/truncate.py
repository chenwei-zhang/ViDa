import re
import os
import shutil
import sys



def truncate_traj(file_name, output_dir, interval):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # List to hold the selected lines
    selected_lines = []
    num_newlines = 0
    
    # Iterate through the lines
    for i, line in enumerate(lines):
        # Check if the line does not contain '(', ')', and '.'
        if '(' not in line and ')' not in line and '.' not in line and line != 'Energy Mismatch\n':
            selected_lines.append(line)
            selected_lines.append(lines[i+1])
            num_newlines += 1
            
            
        # Otherwise, keep lines every [interval] steps
        elif i % interval == 0 and line != 'Energy Mismatch\n':  # Adjust the modulo operation as needed based on which line you want to start with
            selected_lines.append(line)
            num_newlines += 1
            
        # Always include the last line
        if i == len(lines) - 1:
            selected_lines.append(line)
            num_newlines += 1

    # Remove any potential duplicate of the last line
    if len(selected_lines) >= 2 and selected_lines[-1] == selected_lines[-2]:
        selected_lines.pop()
        
    # Write the selected lines to a new file
    base_name = os.path.basename(os.path.splitext(file_name)[0])
    
    with open(f'{output_dir}/{base_name}.txt', 'w') as new_file:
        new_file.writelines(selected_lines)

    return num_newlines




with open('./gentrj.out') as f:
    lines = f.readlines()
    
output_dir = './raw_data/Machinek-PRF'

num_newlines_list = []

for i in range(len(lines)):
    # Regular expression to find digits in the string
    match1 = re.search(r'Simulation (\d+)', lines[i])
    match2 = re.search(r'Trajectory length: (\d+)', lines[i])

    # Extracting the number if found
    if match2:
        num_traj = int(match2.group(1)) 
        file_id = int(match1.group(1))

        file_name = f'./raw_data/Machinek-PRF-og/Machinek-PRF-{file_id}.txt'
        
        if num_traj > 1e6:
            interval = 100
            num_newlines = truncate_traj(file_name, output_dir, interval)
            
        elif 1e5 < num_traj < 1e6:
            interval = 50
            num_newlines = truncate_traj(file_name, output_dir, interval)
        
        else:
            shutil.copy(file_name, f'{output_dir}/Machinek-PRF-{file_id}.txt')
            num_newlines = num_traj
        
    else:
        print("No number found in the string.")
        
    num_newlines_list.append(num_newlines)
    
    
for i in range(len(num_newlines_list)):
    print(f"Simulation {i} complete; Truncated trajectory length: {num_newlines_list[i]}")
    sys.stdout.flush()

    