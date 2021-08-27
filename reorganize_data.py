from pathlib import Path
import os
import shutil
import itertools
import logging
import random
import argparse
import copy
import json

default_config = {
    'source_dir': '/macierz/home/s165115/legosymlink/kzawora/test-dst/train',
    'target_dir': '/macierz/home/s165115/legosymlink/kzawora/test-reorg/train',
    'balance': True,
    'photos_per_class': None,
    'add_missing_classes': True,
    'class_map': {
        'class1': ['822931', '915460'],
        'class2': ['852929', '901078', '853045'],
        'class3': ['966967']
    }
}


def copytree(src, dst, symlinks=False, ignore=None):
    logging.info(f'Copying {src} -> {dst}')
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def balancer_process_single_directory(src: Path):
    items = len([f for f in src.glob('*') if f.is_file()])
    return items


def balancer_delete_images(directory, number_of_images, extension='jpg'):
    images = directory.glob(f'*.{extension}')
    for image in random.sample(list(images), number_of_images):
        image.unlink()


def balancer_get_relative_directories(src: Path):
    dirs = []
    for i in src.rglob('*'):
        if i.is_dir():
            dirs.append(i.relative_to(src))
    return dirs


def balance_recursive_directories(src: Path, target_count=None):
    rel_dirs = balancer_get_relative_directories(src)
    src_dirs = [src / rel_dir for rel_dir in rel_dirs]
    lengths = {}
    for src_dir in src_dirs:
        lengths[src_dir] = balancer_process_single_directory(src_dir)
    target_count = target_count if target_count != None else min(
        lengths.values())
    for src_dir in src_dirs:
        to_delete = lengths[src_dir] - target_count
        logging.info(f'Found {lengths[src_dir]} images in {src_dir}. Removing {to_delete} to meet the target of {target_count} images.')
        if to_delete > 0:
            balancer_delete_images(src_dir, to_delete)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True,
                        help='config json file name', dest='json')
    args = parser.parse_args()
    default_config = {
        'source_dir': '',#'/macierz/home/s165115/legosymlink/kzawora/test-dst/train',
        'target_dir': '',#'/macierz/home/s165115/legosymlink/kzawora/test-reorg/train',
        'balance': False,
        'photos_per_class': None,
        'add_missing_classes': False,
        'class_map': {
            #    'class1': ['822931', '915460'],
            #    'class2': ['852929', '901078', '853045'],
            #    'class3': ['966967']
        }
    }

    init_dict = copy.deepcopy(default_config)
    with open(str(args.json)) as json_file:
        args = json.load(json_file)
        init_dict.update(args)

    classes_in_class_map = set(itertools.chain(
        *init_dict['class_map'].values()))
    classes_in_src = set()
    for i in Path(init_dict['source_dir']).glob('*'):
        if i.is_dir():
            classes_in_src.add(i.stem)

    missing_classes_in_map = classes_in_src - classes_in_class_map
    missing_classes_in_src = classes_in_class_map - classes_in_src

    if missing_classes_in_src != set():
        raise Exception(
            f'Invalid classes found in class map values: {missing_classes_in_src}')

    
    class_map = init_dict['class_map']
    if len(missing_classes_in_map) != 0:
        logging.info(f'Found missing classes in provided map: {missing_classes_in_map}')
    if init_dict['add_missing_classes']:
        logging.info(f'Missing classes WILL be copied since "add_missing_classes" option is enabled.')
        for clazz in missing_classes_in_map:
            class_map[clazz] = [clazz]
    else:
        logging.info(f'Missing classes WILL NOT be copied since "add_missing_classes" option is disabled.')

    def get_dst_path(k): return Path(init_dict['target_dir'])/Path(k)
    def get_src_path(v): return Path(init_dict['source_dir'])/Path(v)
    logging.info(f'Starting data reorganization...')
    for target_class, list_of_classes in init_dict['class_map'].items():
        dst_path = get_dst_path(target_class)
        dst_path.mkdir(exist_ok=True, parents=True)
        for clazz in list_of_classes:
            src_path = get_src_path(clazz)
            copytree(src_path, dst_path)
    logging.info(f'Data reorganization done.')
    if init_dict['balance']:
        logging.info(f'Balancing reorganized data...')
        balance_recursive_directories(Path(init_dict['target_dir']), init_dict['photos_per_class'])
    logging.info(f'All done! Goodbye!')
