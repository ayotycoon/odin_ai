import os

import torch

from main.utils.cacheable import cacheable


@cacheable('1D')
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device

def map_key_from_list(key = '', l = []):
    li =  [getattr(x, key, None) for x in l]
    return li


def get_all_files_recursive(_re_folder_path):
    files = []
    for f in os.listdir(_re_folder_path):
        full_path = os.path.join(_re_folder_path, f)
        if os.path.isfile(full_path):
            files.append(full_path)
        elif os.path.isdir(full_path):
            files.extend(get_all_files_recursive(full_path))  # Recursively call the function for directories
    return files
