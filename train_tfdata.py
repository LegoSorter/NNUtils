from tensorflow.python.keras.callbacks import EarlyStopping
import wandb
import numpy as np
import random
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB4, EfficientNetB7, VGG16, ResNet50V2, InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import logging
import argparse
import time
from tensorflow.keras import mixed_precision
import tensorflow_datasets as tfds

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        #preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        # preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)

class ModelBuilder():
    def __init__(self, cfg):
        self.cfg = cfg

    def generic_builder(self, name, net, lr=1e-2, dropout_rate=0.2):
        cfg = self.cfg
        inputs = layers.Input(shape=cfg['img_shape'])
        x = img_augmentation(inputs)
        if cfg['transfer_learning']:
            model = net(include_top=False, input_tensor=x, weights='imagenet')

            # Freeze the pretrained weights
            model.trainable = False

            # Rebuild top
            x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
            x = layers.BatchNormalization()(x)
            top_dropout_rate = dropout_rate
            x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = layers.Dense(
                cfg['num_classes'], activation="softmax", name="pred")(x)
        else:
            model = net(include_top=False, input_tensor=x, weights=None)
            model.trainable = True
            top_dropout_rate = dropout_rate
            x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = layers.Dense(
                cfg['num_classes'], activation="softmax", name="pred")(x)

        # Compile
        model = tf.keras.Model(inputs, outputs, name=name)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "top_k_categorical_accuracy"]
        )
        return model


class ModelSelector():

    def get_model_map(self):
        return {
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB4': EfficientNetB4,
            'ResNet50': ResNet50,
            'ResNet50V2': ResNet50V2,
            'InceptionV3': InceptionV3
        }

    def __init__(self, cfg):
        self.cfg = cfg
        model_name = self.cfg['model']
        self.mb = ModelBuilder(cfg)
        try:
            model_module = self.get_model_map()[model_name]
            self.model = self.mb.generic_builder(
                model_name, model_module, lr=self.cfg['pre_training_learning_rate'])
        except KeyError:
            raise Exception(f'Invalid model name: {model_name}')

    def get_model(self):
        return self.model

    def unfreeze_top_n_layers(self, n: int, lr=None):
        lr = self.model.optimizer.lr.numpy() if lr is None else lr
        model = self.model
        for layer in model.layers[-n:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "top_k_categorical_accuracy"]
        )
        return model


class WandbTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_time, items_per_epoch):
        self.start_time = start_time
        self.items_per_epoch = items_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_time_start
        wandb.log({
            'epoch_time': epoch_time,
            'total_time': time.time() - self.start_time,
            'fps_per_epoch': self.items_per_epoch/epoch_time
        }, commit=False)
        # self.times.append(epoch_time)
        #self.total_times.append(time.time() - self.start_time)
        # self.fps_per_epoch.append(epoch_time/self.items_per_epoch)


if __name__ == '__main__':
    

    config = {
        'model': 'EfficientNetB0',
        'img_shape': (224, 224, 3),
        'max_epochs': 100,
        'max_epochs_per_fit': 28,
        'num_classes': 447,
        'batch_size': 128,
        'architecture': 'CNN',
        'dataset': 'LEGO_447c',
        'wandb_val_images': 10,
        'transfer_learning': True,
        'pre_training_epochs': 20,
        'pre_training_only': False,
        'pre_training_learning_rate': 1e-2,
        'pre_training_min_delta': 0.01,
        'pre_training_patience': 14,
        'fine_tuning_learning_rate': 1e-4,
        'fine_tuning_min_delta': 0.01,
        'fine_tuning_patience': 5,
        'fine_tuning_unfreeze_interval': 15,
        'dataset_train_dir': '/macierz/home/s165115/legosymlink/kzawora/dataset_new/train',
        'dataset_val_dir': '/macierz/home/s165115/legosymlink/kzawora/dataset_new/val',
        'steps_per_epoch': 1000
    }

    parser = argparse.ArgumentParser(
        description='Sweep train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value),
                            default=value, dest=key, help=f'{key}')
    args = parser.parse_args()
    wandb.init(project='lego447classes', config=args)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    mixed_precision.set_global_policy('mixed_float16')
    cfg = wandb.config
    ms = ModelSelector(cfg)
    model = ms.get_model()
    width, height, depth = cfg['img_shape']
    slow_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        cfg['dataset_train_dir'],
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=cfg['batch_size'],
        image_size=(width, height),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    train_dataset = slow_train_dataset.repeat().shuffle(slow_train_dataset.cardinality().numpy(), reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
                   # tf.data.Dataset.range(2) \
                   #         .interleave(  # Parallelize data reading
                   #             lambda _ : slow_train_dataset,
                   #             num_parallel_calls=tf.data.AUTOTUNE
                   #         ) \
                   #         .prefetch(  # Overlap producer and consumer works
                   #             tf.data.AUTOTUNE
                   #         ) \
    
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        cfg['dataset_val_dir'],
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=cfg['batch_size'],
        image_size=(width, height),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    val_dataset = val_dataset.cache()

    class_names = ['10197', '10201', '10314', '10928', '11090', '11153', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '120493', '131673', '13349', '13547', '13548', '13731', '14395', '14417', '14419', '14704', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15254', '15332', '15379', '15395', '15397', '15460', '15461', '15470', '15535', '15573', '15672', '15706', '15712', '158788', '15967', '16577', '17114', '17485', '18649', '18651', '18653', '18674', '18838', '18969', '19159', '20896', '21229', '216731', '22385', '22388', '22390', '22391', '22885', '22888', '22889', '22890', '22961', '2357', '239356', '24014', '24122', '2412b', '2419', '2420', '242434', '24246', '24299', '24316', '24375', '2441', '2445', '2450', '24505', '2453b', '2454', '2456', '2460', '2476a', '2486', '24866', '2540', '254579', '26047', '2639', '2654', '26601', '26604', '267165', '27255', '27262', '27266', '2730', '2736', '274829', '27940', '2853', '2854', '28653', '2877', '2904', '2926', '292629', '296435', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '30069', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '3020', '3021', '3022', '30237b', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361c', '30363', '30367c', '3037', '3038', '30387', '3040b', '30414', '3045', '30503', '30552', '30553', '30565', '3062', '3068', '3069b', '3070b', '3185', '32000', '32002', '32013', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32062', '32064a', '32073', '32123b', '32124', '32140', '32184', '32187', '32192', '32198', '32250', '32291', '32316', '32348', '3245', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '33291', '33299b', '33909', '3460', '35044', '3622', '3623', '3633', '3639', '3640', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '36840', '36841', '3700', '3701', '3705', '3706', '3707', '3710', '3713', '374125', '3747b', '3795', '3832', '3895', '392043', '3941', '3942c', '3957', '3958', '39739', '4032a', '40490', '4073', '4081b', '4083', '40902', '413097', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '41762', '41768', '41769', '41770', '4185', '42003', '42023', '4216', '4218', '42446', '4274', '4282', '4286', '4287b', '43708', '43712', '43713', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44861', '4488', '4490', '4510', '4519', '45590', '456218', '45677', '4600', '465007', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '474589', '47753', '47755', '47905', '48092', '48171', '48336', '4865b', '4871', '48723', '48729b', '48933', '48989', '496432', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '523081', '52501', '53899', '54383', '54384', '55013', '551028', '56596', '569005', '57519', '57520', '57585', '57895', '57909b', '58090', '59426', '59443', '60032', '6005', '6014', '6020', '60470b', '60471', '60474', '60475b', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60581', '60592', '60593', '60594', '60596', '60598', '60599', '60607', '60608', '60616b', '60621', '60623', '608036', '6081', '60897', '6091', '61070', '61071', '6111', '61252', '612598', '61409', '614655', '61484', '6157', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '64225', '64288', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '77206', '822931', '84954', '85080', '852929', '853045', '85984', '87079', '87081', '87083', '87087', '87544', '87580', '87609', '87620', '87697', '88292', '88323', '88646', '88930', '901078', '90195', '90202', '90609', '90611', '90630', '915460', '92013', '92092', '92582', '92583', '92589', '92907', '92947', '92950', '93273', '93274', '93606', '94161', '959666', '966967', '98100', '98197', '98262', '98283', '98560', '99008', '99021', '99206', '99207', '99773', '99780', '99781']
    wandb_callback = WandbCallback(data_type='image',
                                  # training_data=val_generator[0][:cfg['wandb_val_images']],
                                   labels=class_names, predictions=10)
    items_per_epoch = config['steps_per_epoch'] * \
        config['batch_size'] if config['steps_per_epoch'] is not None else train_dataset.reduce(0, lambda x, _: x + 1).numpy()
    time_callback = WandbTimeCallback(time.time(), items_per_epoch)

    pre_training_early_stopping_callback = EarlyStopping(
        monitor='val_accuracy', mode='max', min_delta=cfg['pre_training_min_delta'], patience=cfg['pre_training_patience'], restore_best_weights=True)
    pre_training_callbacks = [time_callback, wandb_callback,
                              pre_training_early_stopping_callback]

    fine_tuning_early_stopping_callback = EarlyStopping(
        monitor='val_accuracy', mode='max', min_delta=cfg['fine_tuning_min_delta'], patience=cfg['fine_tuning_patience'], restore_best_weights=True)
    fine_tuning_callbacks = [time_callback, wandb_callback,
                             fine_tuning_early_stopping_callback]

    epochs_to_date = 0
    # pre-training
    wandb.log({'trainable_layers': len(
        [1 for layer in model.layers if layer.trainable is True])}, commit=False)
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=cfg['pre_training_epochs'], callbacks=pre_training_callbacks, steps_per_epoch=cfg['steps_per_epoch'])

    # fine-tuning
    if cfg['transfer_learning'] and not cfg['pre_training_only']:
        for to_unfreeze in range(cfg['fine_tuning_unfreeze_interval'], len(model.layers), cfg['fine_tuning_unfreeze_interval']):
            model = ms.unfreeze_top_n_layers(
                to_unfreeze, lr=cfg['fine_tuning_learning_rate'])
            trainable_layers = len(
                [1 for layer in model.layers if layer.trainable is True])
            wandb.log({'trainable_layers': trainable_layers}, commit=False)
            logging.info(f'Fine-tuning on {trainable_layers}')
            epochs_to_date += len(history.history['loss'])
            if epochs_to_date >= cfg['max_epochs']:
                break
            epochs_to_do = epochs_to_date+cfg['max_epochs_per_fit'] if epochs_to_date + \
                cfg['max_epochs_per_fit'] < cfg['max_epochs'] else cfg['max_epochs']
            history = model.fit(train_dataset, validation_data=val_dataset, initial_epoch=epochs_to_date,
                                epochs=epochs_to_do, callbacks=fine_tuning_callbacks, steps_per_epoch=cfg['steps_per_epoch'])
