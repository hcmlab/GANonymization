"""
Created by Fabio Hellmann.
"""

import os
import os.path
import pathlib
import shutil
from glob import glob
from typing import Optional, List, Tuple

from tqdm import tqdm


def move_files(files: List[str], output_dir: str):
    """
    Move a list of files to another directory.
    @param files: Files to move.
    @param output_dir: The output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_path in tqdm(files, desc=output_dir):
        shutil.copyfile(file_path, os.path.join(output_dir, pathlib.Path(file_path).name))


def get_last_ckpt(ckpt_directory: str) -> Optional[str]:
    """
    Retrieve the last checkpoint based on the creation timestamp.
    @param ckpt_directory: The directory where the checkpoints are saved.
    @return: The checkpoint or None if none was found.
    """
    ckpt_files = glob(os.path.join(ckpt_directory, '*.ckpt'))
    if len(ckpt_files) > 0:
        return max(ckpt_files, key=os.path.getctime)
    return None


def glob_dir(directory: str, exclude: Tuple[str] = ('Thumbs.db', '.DS_Store')):
    """
    Recursively search and list all files in the directory filtered by the exclusion clause.
    @param directory: The directory to search in.
    @param exclude: The files or file-endings to filter out.
    @return: A list of files.
    """
    return list(
        filter(lambda f: not f.endswith(exclude),
               glob(os.path.join(directory, '**', '*.*'), recursive=True)))
