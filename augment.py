import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import random
import string
from dataclasses import dataclass, asdict

@dataclass
class AugmenterConfig:
    name: str = 'basic'
    extension: str = 'default'
    width: int = 224
    height: int = 224
    pad_mode: str = 'edge'
    position: str = 'center'


def basic_augmenter(cfg: AugmenterConfig):
    return iaa.Sequential([
        iaa.Resize({'shorter-side': 'keep-aspect-ratio', 'longer-side': max(cfg.width, cfg.height)}),
        iaa.PadToFixedSize(width=cfg.width, height=cfg.height, position=cfg.position, pad_mode=cfg.pad_mode, pad_cval=(0, 0))
    ])


def basic_grayscale_augmenter(cfg: AugmenterConfig):
    return iaa.Sequential([
        iaa.Resize({'shorter-side': 'keep-aspect-ratio', 'longer-side': max(cfg.width, cfg.height)}),
        iaa.PadToFixedSize(width=cfg.width, height=cfg.height, position=cfg.position, pad_mode=cfg.pad_mode),
        iaa.Grayscale()
    ])


def grayscale_augmenter(cfg: AugmenterConfig):
    return iaa.Sequential([
        iaa.Resize({'shorter-side': 'keep-aspect-ratio', 'longer-side': max(cfg.width, cfg.height)}),
        iaa.PadToFixedSize(width=cfg.width, height=cfg.height, position=cfg.position, pad_mode=cfg.pad_mode),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode='edge'
        )),
        iaa.SomeOf((0, 5),
                   [
                       # Convert some images into their superpixel representation,
                       # sample between 20 and 200 superpixels per image, but do
                       # not replace all superpixels with their average, only
                       # some of them (p_replace).
                       iaa.Sometimes(0.5,
                                     iaa.Superpixels(
                                         p_replace=(0, 1.0),
                                         n_segments=(20, 200)
                                     )
                                     ),

                       # Blur each image with varying strength using
                       # gaussian blur (sigma between 0 and 3.0),
                       # average/uniform blur (kernel size between 2x2 and 7x7)
                       # median blur (kernel size between 3x3 and 11x11).
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.AverageBlur(k=(2, 7)),
                           iaa.MedianBlur(k=(3, 11)),
                       ]),

                       # Sharpen each image, overlay the result with the original
                       # image using an alpha between 0 (no sharpening) and 1
                       # (full sharpening effect).
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                       # Same as sharpen, but for an embossing effect.
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                       # Search in some images either for all edges or for
                       # directed edges. These edges are then marked in a black
                       # and white image and overlayed with the original image
                       # using an alpha of 0 to 0.7.
                       iaa.Sometimes(0.5, iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.7)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.7), direction=(0.0, 1.0)
                           ),
                       ])),

                       # Add gaussian noise to some images.
                       # In 50% of these cases, the noise is randomly sampled per
                       # channel and pixel.
                       # In the other 50% of all cases it is sampled once per
                       # pixel (i.e. brightness change).
                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                       ),

                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                       # them to black) or drop them on an image with 2-5% percent
                       # of the original size, leading to large dropped
                       # rectangles.
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.02, 0.05),
                               per_channel=0.2
                           ),
                       ]),

                       # Invert each image's channel with 5% probability.
                       # This sets each pixel value v to 255-v.
                       iaa.Invert(0.05, per_channel=True),  # invert color channels

                       # Add a value of -10 to 10 to each pixel.
                       iaa.Add((-10, 10), per_channel=0.5),

                       # Change brightness of images (50-150% of original value).
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),

                       # Improve or worsen the contrast of images.
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                       # Convert each image to grayscale and then overlay the
                       # result with the original with random alpha. I.e. remove
                       # colors with varying strengths.
                       iaa.Grayscale(alpha=(0.0, 1.0)),

                       # In some images move pixels locally around (with random
                       # strengths).
                       iaa.Sometimes(0.5,
                                     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                                     ),

                       # In some images distort local areas with varying strength.
                       #iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                   ],
                   # do all of the above augmentations in random order
                   random_order=True)#,
       # iaa.Grayscale()
    ])


class Augmenter():
    # map augmenter to strings
    aug_map = {
        'basic': basic_augmenter,
        'basic_grayscale': basic_grayscale_augmenter,
        'grayscale': grayscale_augmenter,
    }

    def __init__(self, config=AugmenterConfig()):
        self.cfg = config
        try:
            self.augmenter = self.aug_map[self.cfg.name](self.cfg)
        except KeyError:
            raise Exception(f'Invalid augmenter name: {self.cfg.name}')

    def __call__(self, *args, **kwargs):
        return self.augmenter(*args, **kwargs)


    def process_single_directory(self, src: Path, dst: Path, img_target: int):
        files = list(src.iterdir())
        target = img_target if img_target != 0 else len(files) 
        images = []
        if len(files) < target:
            images = files
            for _ in range(target-len(files)):
                images.append(random.choice(files))
        else:
            images = list(random.sample(files, target))
        if False:
            loaded_images = [np.array(imageio.imread(file)) for file in images]
            augmented = self.augmenter(images=loaded_images)
            for aug_img, file in zip(augmented, images):
                dst_filename = f'{file.stem}{file.suffix}'
                dst_path = dst / dst_filename
                while dst_path.exists():
                    suffix = ''.join(random.choice(string.ascii_letters) for _ in range(5))
                    dst_filename = f'{file.stem}_{suffix}{file.suffix}'
                    dst_path = dst / dst_filename
                imageio.imwrite(dst_path, aug_img)
        else:
            batch_size = 16
            for i in range(0, len(images), batch_size):
                loaded_images = [np.array(imageio.imread(file)) for file in images[i:i+batch_size]]
                augmented = self.augmenter(images=loaded_images)
                for aug_img, file in zip(augmented, images[i:i+batch_size]):
                    dst_filename = f'{file.stem}{file.suffix}'
                    dst_path = dst / dst_filename
                    while dst_path.exists():
                        suffix = ''.join(random.choice(string.ascii_letters) for _ in range(5))
                        dst_filename = f'{file.stem}_{suffix}{file.suffix}'
                        dst_path = dst / dst_filename
                    imageio.imwrite(dst_path, aug_img)


    def __get_relative_directories(self, src: Path):
        dirs = []
        for i in src.rglob('*'):
            if i.is_dir():
                dirs.append(i.relative_to(src))
        return dirs

    def process_recursive_directories(self, src: Path, dst: Path, img_target=0, num_workers=4):
        rel_dirs = self.__get_relative_directories(src)
        src_dirs = [src / rel_dir for rel_dir in rel_dirs]
        dst_dirs = [dst / rel_dir for rel_dir in rel_dirs]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for src_dir, dst_dir in zip(src_dirs, dst_dirs):
                dst_dir.mkdir(exist_ok=True, parents=True)
                futures.append(executor.submit(self.process_single_directory, src_dir, dst_dir, img_target))
            for future in futures:
                future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Directory path with images to augment', dest='src')
    parser.add_argument('-o', required=True, help='Output directory for augmented images', dest='dst')
    parser.add_argument('-a', default='basic', help=f'Augmenter name. Available: {list(Augmenter.aug_map.keys())}. Default: basic.', dest='aug_name')
    parser.add_argument('-N', default=0, type=int, help=f'Target number of images per class', dest='img_target')
    parser.add_argument('-W', default=AugmenterConfig.width, type=int, help=f'Target image width', dest='width')
    parser.add_argument('-H', default=AugmenterConfig.height, type=int, help=f'Target image  height', dest='height')
    args = parser.parse_args()
    aug_config = AugmenterConfig()
    aug_config.name = args.aug_name
    aug_config.width = args.width
    aug_config.height = args.height
    aug = Augmenter(aug_config)
    aug.process_recursive_directories(Path(args.src), Path(args.dst), args.img_target)
