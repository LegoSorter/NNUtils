import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class AugmenterConfig():
    EXTENSION = 'default'
    IMG_MULTIPLIER = 1
    WIDTH = 224
    HEIGHT = 224
    PAD_MODE = 'edge'
    POSITION = 'center'


def basic_augmenter(cfg: AugmenterConfig):
    return iaa.Sequential([
        iaa.Resize({'shorter-side': 'keep-aspect-ratio', 'longer-side': max(cfg.WIDTH, cfg.HEIGHT)}),
        iaa.PadToFixedSize(width=cfg.WIDTH, height=cfg.HEIGHT, position=cfg.POSITION, pad_mode=cfg.PAD_MODE)
    ])


def basic_grayscale_augmenter(cfg: AugmenterConfig):
    return iaa.Sequential([
        iaa.Resize({'shorter-side': 'keep-aspect-ratio', 'longer-side': max(cfg.WIDTH, cfg.HEIGHT)}),
        iaa.PadToFixedSize(width=cfg.WIDTH, height=cfg.HEIGHT, position=cfg.POSITION, pad_mode=cfg.PAD_MODE),
        iaa.Grayscale()
    ])


def grayscale_augmenter(cfg: AugmenterConfig):
    return iaa.Sequential([
        iaa.Resize({'shorter-side': 'keep-aspect-ratio', 'longer-side': max(cfg.WIDTH, cfg.HEIGHT)}),
        iaa.PadToFixedSize(width=cfg.WIDTH, height=cfg.HEIGHT, position=cfg.POSITION, pad_mode=cfg.PAD_MODE),
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

    def __init__(self, config=AugmenterConfig(), augmenter_name='basic'):
        self.cfg = config
        try:
            self.augmenter = self.aug_map[augmenter_name](self.cfg)
        except KeyError:
            raise Exception(f'Invalid augmenter name: {augmenter_name}')

    def __call__(self, *args, **kwargs):
        return self.augmenter(*args, **kwargs)


def process_single_directory(src: Path, dst: Path, augmenter: Augmenter):
    for file in src.iterdir():
        image = np.array(imageio.imread(file))
        multiplier = augmenter.cfg.IMG_MULTIPLIER
        augmented = augmenter(images=[image] * multiplier)
        for idx, aug_img in enumerate(augmented):
            dst_filename = f'{file.stem}_{idx}{file.suffix}'
            imageio.imwrite(dst / dst_filename, aug_img)


def get_relative_directories(src: Path):
    dirs = []
    for i in src.rglob('*'):
        if i.is_dir():
            dirs.append(i.relative_to(src))
    return dirs


def process_recursive_directories(src: Path, dst: Path, augmenter: Augmenter):
    rel_dirs = get_relative_directories(src)
    src_dirs = [src / rel_dir for rel_dir in rel_dirs]
    dst_dirs = [dst / rel_dir for rel_dir in rel_dirs]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for src_dir, dst_dir in zip(src_dirs, dst_dirs):
            dst_dir.mkdir(exist_ok=True, parents=True)
            futures.append(executor.submit(process_single_directory, src_dir, dst_dir, augmenter))
        for future in futures:
            future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Directory path with images to augment', dest='src')
    parser.add_argument('-o', required=True, help='Output directory for augmented images', dest='dst')
    parser.add_argument('-a', default='basic', help=f'Augmenter name. Available: {list(Augmenter.aug_map.keys())}. Default: basic.', dest='aug_name')
    parser.add_argument('-N', default=AugmenterConfig.IMG_MULTIPLIER, type=int, help=f'Number of replications per image', dest='IMG_MULTIPLIER')
    parser.add_argument('-W', default=AugmenterConfig.WIDTH, type=int, help=f'Target image width', dest='WIDTH')
    parser.add_argument('-H', default=AugmenterConfig.HEIGHT, type=int, help=f'Target image  height', dest='HEIGHT')
    args = parser.parse_args()
    aug_config = AugmenterConfig()
    aug_config.IMG_MULTIPLIER = args.IMG_MULTIPLIER
    aug_config.WIDTH = args.WIDTH
    aug_config.HEIGHT = args.HEIGHT
    aug = Augmenter(aug_config, args.aug_name)
    process_recursive_directories(Path(args.src), Path(args.dst), aug)
