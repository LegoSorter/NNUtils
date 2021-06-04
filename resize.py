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
from typing import List
from PIL import Image
def get_directory_file_count(src: Path):
    return len([f for f in src.glob('*') if f.is_file()])

def get_relative_directories(src: Path):
    dirs = []
    for i in src.rglob('*'):
        if i.is_dir():
            dirs.append(i.relative_to(src))
    return dirs

def resize_images_in_directory(src, dst, rel_dir, sizes):
    images = src.glob(f'*.jpg')
    for image in images:
        image_path = Path(image)
        img = Image.open(image_path)
        for size in sizes:
            img_new = img.copy()
            dst_dir = dst / f'resized_{size}x{size}' / rel_dir
            dst_dir.mkdir(exist_ok=True, parents=True)
            img_new.thumbnail((size, size), Image.ANTIALIAS)
            img_new.save(dst_dir / image_path.name, 'JPEG')
            img_new.close()
        img.close()


def resize_dataset(src: Path, dst: Path, sizes: List[int], prefix=Path('')):
    rel_dirs = get_relative_directories(src)
    src_dirs = [src / rel_dir for rel_dir in rel_dirs]
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for src_dir, rel_dir in zip(src_dirs,rel_dirs):
            futures.append(executor.submit(resize_images_in_directory, src_dir, dst, prefix/rel_dir, sizes))
        for future in futures:
            future.result()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Directory path with images to augment', dest='src')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', help='Output directory', dest='dst')
    parser.add_argument('-l', default=[], nargs='+', dest="size_list")
    args = parser.parse_args()
    resize_dataset(Path(args.src), Path(args.dst), list(map(int,args.size_list)))

