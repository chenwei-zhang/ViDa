import re
import os
import shutil
import sys



def truncate_mismatch(file_name, output_dir, name_id):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # List to hold the selected lines
    selected_lines = []
    num_newlines = 0
    
    # Iterate through the lines
    for line in lines:
        if line != 'Energy Mismatch\n':
            selected_lines.append(line)
            num_newlines += 1
    
    with open(f'{output_dir}/Machinek-PRF-{name_id}.txt', 'w') as new_file:
        new_file.writelines(selected_lines)

    return num_newlines




with open('./trajOriginal.out') as f:
    lines = f.readlines()
    
output_dir = './raw_data/Machinek-PRF-selc'

name_id = 0

for i in range(len(lines)):
    # Regular expression to find digits in the string
    match1 = re.search(r'Simulation (\d+)', lines[i])
    match2 = re.arch(r'Trajectory length: (\d+)', lines[i])
    
    # Extracting the number if found
    if match2:
        num_traj = int(match2.group(1)) 
        file_id = int(match1.group(1))
        
        file_name = f'./raw_data/Machinek-PRF-og/Machinek-PRF-{file_id}.txt'
        
        if num_traj > 5e5:
            continue
            
        else:
            num_newlines = truncate_mismatch(file_name, output_dir, name_id)
            
            print(f"Simulation {i} ---> Simulation {name_id}; Selected trajectory length: {num_newlines}")
            sys.stdout.flush()
            
            name_id += 1
            
    else:
        print("No number found in the string.")
        
    