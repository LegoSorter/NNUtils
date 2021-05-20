import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import random

def process_single_directory(src: Path):
    items = len([f for f in src.glob('*') if f.is_file()])
    logging.error(f'{src}: {items} items')
    return items

def get_relative_directories(src: Path):
    dirs = []
    for i in src.rglob('*'):
        if i.is_dir():
            dirs.append(i.relative_to(src))
    return dirs

def delete_images(directory, number_of_images, extension='png'):
    images = directory.glob(f'*.{extension}')
    for image in random.sample(list(images), number_of_images):
        image.unlink()

def process_recursive_directories(src: Path):
    rel_dirs = get_relative_directories(src)
    src_dirs = [src / rel_dir for rel_dir in rel_dirs]
    lengths = {}
    for src_dir in src_dirs:
        lengths[src_dir] = process_single_directory(src_dir)
    min_val = min(lengths.values())
    print(f'min_val: {min(lengths.values())} | max_val: {max(lengths.values())} | max as total percentage: {list(lengths.values()).count(max(lengths.values()))/len(lengths.values())*100:.2f}% | max count: {list(lengths.values()).count(max(lengths.values()))}')
#    for src_dir in src_dirs:
#        to_delete = lengths[src_dir] - min_val
#        if to_delete > 0:
#            delete_images(src_dir, to_delete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Directory path with images to augment', dest='src')
    args = parser.parse_args()
    process_recursive_directories(Path(args.src))
