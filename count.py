def count_lines_in_txt(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        return len(lines)

# 示例用法
txt_file = 'datasets/WideIRSTD/img_idx/test_WideIRSTD.txt'  # 替换成你的txt文件路径
line_count = count_lines_in_txt(txt_file)
print(f'The number of lines in the txt file is: {line_count}')

import os

def count_files_in_folder(folder_path):
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

# 示例用法
folder_path = 'results/WideIRSTD/mask'  # 替换成你的文件夹路径
file_count = count_files_in_folder(folder_path)
print(f'The number of files in the folder is: {file_count}')
