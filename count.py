def count_lines_in_txt(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        return len(lines)

txt_file = 'datasets/WideIRSTD/img_idx/test_WideIRSTD.txt'
line_count = count_lines_in_txt(txt_file)
print(f'The number of lines in the txt file is: {line_count}')

import os

def count_files_in_folder(folder_path):
    file_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            file_count += 1
    return file_count

# folder_path = 'results/WideIRSTD/ISTDU-Net'
folder_path = 'results/WideIRSTD/mask'
file_count = count_files_in_folder(folder_path)
print(f'The number of files in the folder is: {file_count}')
