import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import preprocessing
from tensorflow.keras import Sequential

MODEL_OUTPUT_DIR="captcha-model"
BATCH_SIZE = 128
COLOR_MODE = "rgba"
IMG_HEIGHT = 50
IMG_WIDTH = 180
CLASS_NO = 23
EPOCHS = 120

train_ds = preprocessing.image_dataset_from_directory(
  directory="data/train",
  label_mode="categorical",
  seed=123,
  color_mode=COLOR_MODE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

validation_ds = preprocessing.image_dataset_from_directory(
  directory="data/validation",
  label_mode="categorical",
  seed=123,
  color_mode=COLOR_MODE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# Configure the dataset for performance
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
validation_ds = configure_for_performance(validation_ds)

def get_model():
    m = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),

      layers.Conv2D(32, (3, 3), padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      layers.Conv2D(64, (3, 3), padding="same"),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      layers.Conv2D(128, (3, 3), padding="same"),
      layers.Conv2D(64, (1, 1), padding="same"),
      layers.Conv2D(128, (3, 3), padding="same"),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      layers.Conv2D(256, (3, 3), padding="same"),
      layers.Conv2D(128, (1, 1), padding="same"),
      layers.Conv2D(256, (3, 3), padding="same"),
      layers.Conv2D(128, (1, 1), padding="same"),
      layers.Conv2D(256, (3, 3), padding="same"),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      layers.Conv2D(512, (3, 3), padding="same"),
      layers.Conv2D(256, (1, 1), padding="same"),
      layers.Conv2D(512, (3, 3), padding="same"),
      layers.Conv2D(256, (1, 1), padding="same"),
      layers.Conv2D(512, (3, 3), padding="same"),
      layers.Conv2D(256, (1, 1), padding="same"),
      layers.Conv2D(512, (3, 3), padding="same"),
      layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
      layers.Flatten(),
      layers.Dropout(rate=0.5),
      layers.Dense(CLASS_NO, activation="softmax"),
    ])
    m.compile(
      loss=losses.CategoricalCrossentropy(),
      metrics=['accuracy'],
      optimizer=optimizers.Adam(learning_rate=0.00001),
    )
    return m

model=get_model()
model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds, verbose=1)
model.summary()
model.save(MODEL_OUTPUT_DIR)