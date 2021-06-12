from tensorflow.python.keras.callbacks import EarlyStopping
import wandb
import numpy as np
import random
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow.keras.applications as apps
import os
import logging
import argparse
import time
from tensorflow.keras import mixed_precision
import json
import copy

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ModelSelector():
    def get_model_map(self):
        return {
            'DenseNet121': apps.DenseNet121,
            'DenseNet169': apps.DenseNet169,
            'DenseNet201': apps.DenseNet201,
            'EfficientNetB0': apps.EfficientNetB0,
            'EfficientNetB1': apps.EfficientNetB1,
            'EfficientNetB2': apps.EfficientNetB2,
            'EfficientNetB3': apps.EfficientNetB3,
            'EfficientNetB4': apps.EfficientNetB4,
            'EfficientNetB5': apps.EfficientNetB5,
            'EfficientNetB6': apps.EfficientNetB6,
            'EfficientNetB7': apps.EfficientNetB7,
            'InceptionResNetV2': apps.InceptionResNetV2,
            'InceptionV3': apps.InceptionV3,
            'MobileNet': apps.MobileNet,
            'MobileNetV2': apps.MobileNetV2,
            'MobileNetV3Large': apps.MobileNetV3Large,
            'MobileNetV3Small': apps.MobileNetV3Small,
            'NasNetLarge': apps.NASNetLarge,
            'NasNetMobile': apps.NASNetMobile,
            'ResNet101': apps.ResNet101,
            'ResNet101V2': apps.ResNet101V2,
            'ResNet152': apps.ResNet152,
            'ResNet152V2': apps.ResNet152V2,
            'ResNet50': apps.ResNet50,
            'ResNet50V2': apps.ResNet50V2,
            'VGG16': apps.VGG16,
            'VGG19': apps.VGG19,
            'Xception': apps.Xception
        }

    def get_preprocessor_map(self):
        return {
            'DenseNet121': apps.densenet.preprocess_input,
            'DenseNet169': apps.densenet.preprocess_input,
            'DenseNet201': apps.densenet.preprocess_input,
            'EfficientNetB0': apps.efficientnet.preprocess_input,
            'EfficientNetB1': apps.efficientnet.preprocess_input,
            'EfficientNetB2': apps.efficientnet.preprocess_input,
            'EfficientNetB3': apps.efficientnet.preprocess_input,
            'EfficientNetB4': apps.efficientnet.preprocess_input,
            'EfficientNetB5': apps.efficientnet.preprocess_input,
            'EfficientNetB6': apps.efficientnet.preprocess_input,
            'EfficientNetB7': apps.efficientnet.preprocess_input,
            'InceptionResNetV2': apps.inception_resnet_v2.preprocess_input,
            'InceptionV3': apps.inception_v3.preprocess_input,
            'MobileNet': apps.mobilenet.preprocess_input,
            'MobileNetV2': apps.mobilenet_v2.preprocess_input,
            'MobileNetV3Large': apps.mobilenet_v3.preprocess_input,
            'MobileNetV3Small': apps.mobilenet_v3.preprocess_input,
            'NasNetLarge': apps.nasnet.preprocess_input,
            'NasNetMobile': apps.nasnet.preprocess_input,
            'ResNet101': apps.resnet.preprocess_input,
            'ResNet101V2': apps.resnet_v2.preprocess_input,
            'ResNet152': apps.resnet.preprocess_input,
            'ResNet152V2': apps.resnet_v2.preprocess_input,
            'ResNet50': apps.resnet.preprocess_input,
            'ResNet50V2': apps.resnet_v2.preprocess_input,
            'VGG16': apps.vgg16.preprocess_input,
            'VGG19': apps.vgg19.preprocess_input,
            'Xception': apps.xception.preprocess_input
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

    def get_preprocessing_function(self):
        return self.get_preprocessor_map()[self.cfg['model']]

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
    def __init__(self, start_time, items_per_epoch, time_limit=None):
        self.start_time = start_time
        self.items_per_epoch = items_per_epoch
        self.total_time = 0
        self.time_limit = time_limit
        self.exceeded_time_limit = False

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.total_time = time.time() - self.start_time
        epoch_time = time.time() - self.epoch_time_start
        wandb.log({
            'epoch_time': epoch_time,
            'total_time': self.total_time,
            'fps_per_epoch': self.items_per_epoch/epoch_time
        }, commit=False)
        if self.time_limit is not None and self.total_time > self.time_limit:
            logging.info(f'Total training time ({self.total_time:.2f} s) has exceeded the time limit (({self.time_limit:.2f} s). Stopping training.')
            self.model.stop_training = True
            self.exceeded_time_limit = True

    def has_exceeded_time_limit(self):
        return self.exceeded_time_limit
        # self.times.append(epoch_time)
        #self.total_times.append(time.time() - self.start_time)
        # self.fps_per_epoch.append(epoch_time/self.items_per_epoch)


if __name__ == '__main__':

    default_config = {
        'project': 'legotest',
        'name': 'EfficientNetB0-test',
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
        'steps_per_epoch': 100,
        'optimizer': 'sgd',
        'time_limit': None
    }
    # with open('result.json', 'w') as fp:
    #    json.dump(config, fp)
    # exit(0)
    parser = argparse.ArgumentParser(
        description='Sweep train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(f'--json', type=str, nargs='+',
                        default=None, dest='json', help=f'json config filename. Ignores other arguments if provided.')
    for key, value in default_config.items():
        if type(value) is tuple:
            parser.add_argument(f'--{key}', type=type(value[0]),
                                default=value, dest=key, nargs='+', help=f'{key}')
        else:
            parser.add_argument(f'--{key}', type=type(value),
                                default=value, dest=key, help=f'{key}')
    args = parser.parse_args()

    init_dict = copy.deepcopy(default_config)
    if args.json is not None:
        for json_path in args.json:
            with open(json_path) as json_file:
                args = json.load(json_file)
                init_dict.update(args)
    else:
        del args.json
        init_dict = vars(args)

    wandb.init(project=init_dict['project'], name=init_dict['name'], config=init_dict)
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
    train_datagen = ImageDataGenerator(preprocessing_function=ms.get_preprocessing_function())
    #train_datagen = ImageDataGenerator() if cfg['model'][:-1] == 'EfficientNetB' else ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        # '/macierz/home/s165115/legosymlink/kzawora/dataset_processed/train',
        cfg['dataset_train_dir'],
        target_size=(width, height),
        batch_size=cfg['batch_size'], shuffle=True)
    val_datagen = ImageDataGenerator(preprocessing_function=ms.get_preprocessing_function())
    #val_datagen = ImageDataGenerator() if cfg['model'] == 'EfficientNetB0' else ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        cfg['dataset_val_dir'],
        target_size=(width, height),
        batch_size=cfg['batch_size'], shuffle=True)
    wandb_callback = WandbCallback(data_type='image',
                                   # training_data=val_generator[0][:cfg['wandb_val_images']],
                                   labels=list(train_generator.class_indices.keys()), predictions=10)
    items_per_epoch = cfg['steps_per_epoch'] * cfg['batch_size'] if cfg['steps_per_epoch'] is not None else train_generator.samples
    logging.info(f'Images per epoch: {items_per_epoch}')
    tl = cfg['time_limit']
    logging.info(f'Time limit set to {tl:.2f} seconds')

    time_callback = WandbTimeCallback(time.time(), items_per_epoch, time_limit=cfg['time_limit'])

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
    if cfg['transfer_learning'] and not cfg['pre_training_only'] and not time_callback.has_exceeded_time_limit():
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
            if time_callback.has_exceeded_time_limit():
                break
