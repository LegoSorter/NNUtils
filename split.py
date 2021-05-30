import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import random
import shutil
import math
import logging

def get_directory_file_count(src: Path):
    return len([f for f in src.glob('*') if f.is_file()])

def get_relative_directories(src: Path):
    dirs = []
    for i in src.rglob('*'):
        if i.is_dir():
            dirs.append(i.relative_to(src))
    return dirs

def move_images(directory, number_of_images, dst, extension='jpg'):
    images = directory.glob('*')
    for image in random.sample(list(images), number_of_images):
        image.rename(dst / image.name)

def split_dataset(src: Path, n: int, dst1: Path, dst2: Path):
    rel_dirs = get_relative_directories(src)
    shutil.copytree(src, dst1, dirs_exist_ok=True)
    src_dirs = [dst1 / rel_dir for rel_dir in rel_dirs]
    dst_dirs = [dst2 / rel_dir for rel_dir in rel_dirs]
    lengths = {}
    for src_dir in src_dirs:
        lengths[src_dir] = get_directory_file_count(src_dir)
    logging.debug(f'src count: {min(lengths.values())}')
 #   min_val = min(lengths.values())
    for src_dir, dst_dir in zip(src_dirs,dst_dirs):
        dst_dir.mkdir(exist_ok=True, parents=True)
        to_move = lengths[src_dir] - n
        logging.debug(to_move)
        if to_move > 0:
            move_images(src_dir, to_move, dst_dir)

    lengths = {}
    for src_dir in src_dirs:
        lengths[src_dir] = get_directory_file_count(src_dir)
    logging.debug(f'dst1 count: {min(lengths.values())}')

    lengths = {}
    for src_dir in dst_dirs:
        lengths[src_dir] = get_directory_file_count(src_dir)
    logging.debug(f'dst2 count: {min(lengths.values())}')


def split_dataset_by_ratio(src: Path, ratio: float, dst1: Path, dst2: Path):
    assert ratio <= 1 
    rel_dirs = get_relative_directories(src)
    shutil.copytree(src, dst1, dirs_exist_ok=True)
    src_dirs = [dst1 / rel_dir for rel_dir in rel_dirs]
    dst_dirs = [dst2 / rel_dir for rel_dir in rel_dirs]
    lengths = {}
    for src_dir in src_dirs:
        lengths[src_dir] = get_directory_file_count(src_dir)
    logging.debug(f'src counts: {lengths.values()}')
 #   min_val = min(lengths.values())
    for src_dir, dst_dir in zip(src_dirs,dst_dirs):
        dst_dir.mkdir(exist_ok=True, parents=True)
        to_move = math.ceil(lengths[src_dir] * (1-ratio))
        if to_move == lengths[src_dir]:
            to_move -= 1
 #       logging.debug(to_move)
        assert to_move > 0
        if to_move > 0:
            move_images(src_dir, to_move, dst_dir)
    
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return

    lengths = {}
    for src_dir in src_dirs:
        lengths[src_dir] = get_directory_file_count(src_dir)
    logging.debug(f'dst1 counts: {lengths.values()}')

    lengths = {}
    for src_dir in dst_dirs:
        lengths[src_dir] = get_directory_file_count(src_dir)
    logging.debug(f'dst2 counts: {lengths.values()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Directory path with images to augment', dest='src')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', help='Number to split', dest='n')
    group.add_argument('-r', help='Ratio to split', dest='ratio')
    
    parser.add_argument('-o1', required=True, help='output (1-n)', dest='dst1')
    parser.add_argument('-o2', required=True, help='output (1-n)', dest='dst2')


    args = parser.parse_args()

    if args.n:
        split_dataset(Path(args.src), int(args.n), Path(args.dst1), Path(args.dst2))
    elif args.ratio:
        split_dataset_by_ratio(Path(args.src), float(args.ratio), Path(args.dst1), Path(args.dst2))

