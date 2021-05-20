import json
from pathlib import Path
import dacite as dc
import shutil
from augment import Augmenter, AugmenterConfig
from split import split_dataset
import logging
from dataclasses import dataclass, asdict
from typing import List
        
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
    keep_copies: bool
    
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
            self.augmenter.process_recursive_directories(dataset.src, dataset.dst / 'aug', img_target, self.cfg.cpu_threads)
    def split_datasets(self):
        for dataset in self.cfg.datasets:
            split_dataset(dataset.dst / 'aug', dataset.train_target, dataset.dst / 'train', dataset.dst / 'val')
            if not self.cfg.keep_copies:
                shutil.rmtree(dataset.dst / 'aug')
    def merge_datasets(self):
        for dataset in self.cfg.datasets:
            shutil.move(dataset.dst / 'train', self.cfg.dst / 'train')
            shutil.move(dataset.dst / 'val', self.cfg.dst / 'val')
    def process(self):
        logging.error('Augmenting...')
        self.augment_datasets()
        logging.error('Augmenting done! Splitting...')
        self.split_datasets()
        logging.error('Splitting done! Merging...')
        self.merge_datasets()
        logging.error('Merging done! Have a good day!')

if __name__ == '__main__':
    x = Path('config.json').read_text()
    dataset_config = dc.from_dict(data_class=DatasetConfig, data=json.loads(x), config=dc.Config(cast=[Path]))
    dm = DatasetManager(dataset_config)
    dm.process()