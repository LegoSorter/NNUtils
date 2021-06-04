import json
from pathlib import Path
import dacite as dc
import shutil
from augment import Augmenter, AugmenterConfig
from split import split_dataset, split_dataset_by_ratio
from resize import resize_dataset
import logging
from dataclasses import dataclass, asdict
from typing import List
import argparse
import random
import logging
import string


@dataclass
class DatasetSource:
    name: str
    src: Path
    dst: Path
    train_target: int
    val_target: int


@dataclass
class DatasetConfig:
    datasets: List[DatasetSource]
    augmenter_config: AugmenterConfig
    cpu_threads: int
    dst: Path
    delete_individual_dsts: bool
    split_augment_merge: bool
    create_resized_versions: bool
    size_list: List[int]


def copy_wrapper(src, dst, *args, **kwargs):
    dst_path = Path(dst)
    if not dst_path.is_file():
        return shutil.copy2(src, dst, *args, **kwargs)
    while dst_path.exists():
        logging.debug(f'CONFLICT DETECTED: {src} -> {dst}')
        suffix = ''.join(random.choice(string.ascii_letters) for _ in range(5))
        dst_filename = f'{dst_path.stem}_{suffix}{dst_path.suffix}'
        dst_path = dst_path.parent / dst_filename
    return shutil.copy2(src, dst_path, *args, **kwargs)


class DatasetManager():
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.augmenter = Augmenter(cfg.augmenter_config)

    def copy_datasets(self):
        for dataset in self.cfg.datasets:
            shutil.copytree(dataset.src, dataset.dst, dirs_exist_ok=True)

    def augment_datasets(self):
        for dataset in self.cfg.datasets:
            img_target = dataset.train_target + dataset.val_target
            self.augmenter.process_recursive_directories(
                dataset.src, dataset.dst / 'aug', img_target, self.cfg.cpu_threads)

    def split_datasets(self):
        for dataset in self.cfg.datasets:
            split_dataset(dataset.dst / 'aug', dataset.train_target,
                          dataset.dst / 'train', dataset.dst / 'val')
            if not self.cfg.delete_individual_dsts:
                shutil.rmtree(dataset.dst / 'aug')

    def merge_datasets(self):
        for dataset in self.cfg.datasets:
            shutil.move(dataset.dst / 'train', self.cfg.dst / 'train')
            shutil.move(dataset.dst / 'val', self.cfg.dst / 'val')

    def process_asm(self):
        self.augment_datasets()
        logging.info('Augmenting done! Splitting...')
        self.split_datasets()
        logging.info('Splitting done! Merging...')
        self.merge_datasets()
        logging.info('Merging done! Have a good day!')

    def process_sam(self):
        logging.info('SAM (split-augment-merge) processing...')
        for dataset in self.cfg.datasets:
            ratio = dataset.train_target / (dataset.train_target + dataset.val_target)
            logging.info(f'{dataset.name} dataset train/val ratio: {ratio}')
            split_dataset_by_ratio(dataset.src, ratio, dataset.dst / 'train', dataset.dst / 'val')
        logging.info('Splitting done! Augmenting...')

        for dataset in self.cfg.datasets:
            self.augmenter.process_recursive_directories(
                dataset.src, dataset.dst / 'aug' / 'train', dataset.train_target, self.cfg.cpu_threads)
            self.augmenter.process_recursive_directories(
                dataset.src, dataset.dst / 'aug' / 'val', dataset.val_target, self.cfg.cpu_threads)
        logging.info('Augmenting done! Merging...')

        dst = self.cfg.dst / 'train'  if not self.cfg.create_resized_versions else self.cfg.dst / f'orig_{self.cfg.augmenter_config.width}x{self.cfg.augmenter_config.height}'
        for dataset in self.cfg.datasets:
            shutil.copytree(dataset.dst / 'aug' / 'train', dst / 'train', 
                            dirs_exist_ok=True, copy_function=copy_wrapper)
            shutil.copytree(dataset.dst / 'aug' / 'val', dst / 'val', 
                            dirs_exist_ok=True, copy_function=copy_wrapper)
        
        if self.cfg.create_resized_versions:
            logging.info('Merging done! Resizing...')
            sizes = list(map(int, self.cfg.size_list))
            resize_dataset(dst / 'train', self.cfg.dst, sizes, prefix='train')
            resize_dataset(dst / 'val', self.cfg.dst, sizes, prefix='val')
            logging.info('Resizing done! Cleaning up...')
        else:
            logging.info('Merging done! Cleaning up...')
        
        if not self.cfg.delete_individual_dsts:
            logging.info('Merging/resizing done! Have a good day!')
            return
            
        for dataset in self.cfg.datasets:
            shutil.rmtree(dataset.dst)
        logging.info('Cleaning done! Have a good day!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True,
                        help='config json file name', dest='src')
    args = parser.parse_args()

    x = Path(args.src).read_text()
    dataset_config = dc.from_dict(
        data_class=DatasetConfig, data=json.loads(x), config=dc.Config(cast=[Path]))
    dm = DatasetManager(dataset_config)
    dm.process_sam()
