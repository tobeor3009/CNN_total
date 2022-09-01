import os
from glob import glob


def remove_png_files_in_folder(folder_path):
    regxp_files = os.path.join(folder_path, "*.png")
    target_files = glob(regxp_files)
    for target_file in target_files:
        os.remove(target_file)


def get_filebasename(file_path):
    filename_with_ext = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename_with_ext)[0]

    return filename_without_ext
