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
import json

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        try:
            model_module = self.get_model_map()[model_name]
            self.model = self.generic_builder(
                model_name, model_module, lr=self.cfg['pre_training_learning_rate'])
        except KeyError:
            raise Exception(f'Invalid model name: {model_name}')

    def get_model(self):
        return self.model

    def get_optimizer(self, learning_rate):
        opt_name = self.cfg['optimizer']
        if opt_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if opt_name == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        raise Exception(f'Invalid optimizer name: {opt_name}')

    def unfreeze_top_n_layers(self, n: int, lr=None):
        lr = self.model.optimizer.lr.numpy() if lr is None else lr
        model = self.model
        for layer in model.layers[-n:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        optimizer = self.get_optimizer(learning_rate=lr)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", "top_k_categorical_accuracy"]
        )
        return model

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
        optimizer = self.get_optimizer(learning_rate=lr)
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
        'project': 'lego447classes',
        'name': 'EfficientNetB0-run',
        'model': 'EfficientNetB0',
        'img_shape': (224, 224, 3),
        'max_epochs': 200,
        'max_epochs_per_fit': 50,
        'num_classes': 447,
        'batch_size': 128,
        'architecture': 'CNN',
        'dataset': 'LEGO_447c',
        'wandb_val_images': 10,
        'transfer_learning': True,
        'pre_training_epochs': 25,
        'pre_training_only': False,
        'pre_training_learning_rate': 1e-2,
        'pre_training_min_delta': 0.01,
        'pre_training_patience': 5,
        'fine_tuning_learning_rate': 1e-4,
        'fine_tuning_min_delta': 0.01,
        'fine_tuning_patience': 5,
        'fine_tuning_unfreeze_interval': 15,
        'dataset_train_dir': '/macierz/home/s165115/legosymlink/kzawora/dataset_new/train',
        'dataset_val_dir': '/macierz/home/s165115/legosymlink/kzawora/dataset_new/val',
        'steps_per_epoch': None,
        'optimizer': 'sgd'
    }
    # with open('result.json', 'w') as fp:
    #    json.dump(config, fp)
    # exit(0)
    parser = argparse.ArgumentParser(
        description='Sweep train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(f'--json', type=str,
                        default=None, dest='json', help=f'json config filename. Ignores other arguments if provided.')
    for key, value in config.items():
        if type(value) is tuple:
            parser.add_argument(f'--{key}', type=type(value[0]),
                                default=value, dest=key, nargs='+', help=f'{key}')
        else:
            parser.add_argument(f'--{key}', type=type(value),
                                default=value, dest=key, help=f'{key}')
    args = parser.parse_args()
    if args.json is not None:
        with open(args.json) as json_file:
            args = json.load(json_file)
    else:
        del args.json
    wandb.init(config=args)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # tensorflow setup
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
    mixed_precision.set_global_policy('mixed_float16')

    cfg = wandb.config
    ms = ModelSelector(cfg)
    model = ms.get_model()
    width, height, depth = cfg['img_shape']
    train_datagen = ImageDataGenerator(
    ) if cfg['model'] == 'EfficientNetB0' else ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        # '/macierz/home/s165115/legosymlink/kzawora/dataset_processed/train',
        cfg['dataset_train_dir'],
        target_size=(width, height),
        batch_size=cfg['batch_size'], shuffle=True)
    val_datagen = ImageDataGenerator(
    ) if cfg['model'] == 'EfficientNetB0' else ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        cfg['dataset_val_dir'],
        target_size=(width, height),
        batch_size=cfg['batch_size'], shuffle=True)
    wandb_callback = WandbCallback(data_type='image',
                                   # training_data=val_generator[0][:cfg['wandb_val_images']],
                                   labels=list(train_generator.class_indices.keys()), predictions=10)
    items_per_epoch = config['steps_per_epoch'] * \
        config['batch_size'] if config['steps_per_epoch'] is not None else train_generator.samples
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
    history = model.fit(train_generator, validation_data=val_generator,
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
            history = model.fit(train_generator, validation_data=val_generator, initial_epoch=epochs_to_date,
                                epochs=epochs_to_do, callbacks=fine_tuning_callbacks, steps_per_epoch=cfg['steps_per_epoch'])
