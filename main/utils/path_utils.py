import os


def safe_access_path(file_path:str):
    has_ext = len(file_path.split('/')[-1].split('.')) == 2
    if has_ext:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else:
        os.makedirs(file_path, exist_ok=True)
    return file_path

def replace_spaces_in_path(path, replacement="_"):

    path_parts = path.split(os.sep)  # Split the path into parts
    modified_parts = [part.replace(" ", replacement) for part in path_parts]  # Replace spaces in folder names
    return os.sep.join(modified_parts)  # Reconstruct the path