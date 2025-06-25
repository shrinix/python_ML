import numpy as np
import tensorflow as tf
import datetime

import tensorflow as tf
print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
tf.print(physical_devices)
#tf.config.set_visible_devices([], "GPU")
try:
  # Disable all CPUS
  tf.config.set_visible_devices([], 'GPU')
  visible_devices = tf.config.get_visible_devices()
  for device in visible_devices:
    assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
  pass

x = np.random.random((10000, 5))
y = np.random.random((10000, 2))

x2 = np.random.random((2000, 5))
y2 = np.random.random((2000, 2))

inp = tf.keras.layers.Input(shape = (5,))
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(inp)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
o = tf.keras.layers.Dense(2, activation = 'sigmoid')(l1)

model = tf.keras.models.Model(inputs = [inp], outputs = [o])
model.compile(optimizer = "Adam", loss = "mse")
train_start_time = datetime.datetime.now()
print("Training start time: ", datetime.datetime.now())
model.fit(x, y, validation_data = (x2, y2), batch_size = 500, epochs = 500)
train_end_time = datetime.datetime.now()
print("Total training time: ", train_end_time - train_start_time)
print("Training end time: ", datetime.datetime.now())