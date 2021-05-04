from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB4, EfficientNetB7, VGG16, ResNet50V2, InceptionV3
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class Config():
    IMG_SIZE = 224
    NUM_CLASSES = 6
    MODEL = 'ResNet50V2'
    TRANSFER_LEARNING = True
    ONLINE_AUGMENTATION = True
    BATCH_SIZE = 8

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        #preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        #preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


class ModelBuilder():
    def __init__(self, cfg):
        self.cfg = cfg

    def __generic_builder(self, name, net, lr=1e-2, dropout_rate=0.2):
        cfg = self.cfg
        inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
        x = img_augmentation(inputs) if cfg.ONLINE_AUGMENTATION else inputs
        if cfg.TRANSFER_LEARNING:
            model = net(include_top=False, input_tensor=x, weights='imagenet')

            # Freeze the pretrained weights
            model.trainable = False

            # Rebuild top
            x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
            x = layers.BatchNormalization()(x)
            top_dropout_rate = 0.2
            x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = layers.Dense(cfg.NUM_CLASSES, activation="softmax", name="pred")(x)
        else:
            model = net(include_top=False, input_tensor=x, weights=None)
            model.trainable = True
            top_dropout_rate = 0.2
            x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = layers.Dense(cfg.NUM_CLASSES, activation="softmax", name="pred")(x)

        # Compile
        model = tf.keras.Model(inputs, outputs, name=name)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def build_efficientnetb0(self):
        return self.__generic_builder('EfficientNetB0', EfficientNetB0)

    def build_resnet50v2(self):
        return self.__generic_builder('ResNet50V2', ResNet50V2)

    def build_resnet50(self):
        return self.__generic_builder('ResNet50', ResNet50)

    def build_inceptionv3(self):
        return self.__generic_builder('InceptionV3', InceptionV3)


class ModelSelector():

    def get_model_map(self):
        return {
            'EfficientNetB0': self.mb.build_efficientnetb0,
            'ResNet50': self.mb.build_resnet50,
            'ResNet50V2': self.mb.build_resnet50v2,
            'InceptionV3': self.mb.build_inceptionv3
        }

    def __init__(self, cfg):
        self.cfg = cfg
        model_name = self.cfg.MODEL
        self.mb = ModelBuilder(cfg)
        try:
            self.model = self.get_model_map()[model_name]()
        except KeyError:
            raise Exception(f'Invalid model name: {model_name}')

    def get_model(self):
        return self.model

    def unfreeze_model(self):
        model = self.model
        for layer in model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model



if __name__ == '__main__':
    cfg = Config()
    ms = ModelSelector(cfg)
    model = ms.get_model()
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        '/home/konrad/dev/mgr/dev_dataset_small_color',
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE)

    model.fit(train_generator, epochs=2)

    if cfg.TRANSFER_LEARNING:
        model = ms.unfreeze_model()
        model.fit(train_generator, epochs=2)