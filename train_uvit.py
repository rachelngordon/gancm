import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
from uvit import uvit
from uvit import modules_uvit as modules


x_data_path = "/media/aisec-102/DATA3/rachel/data/CV/normalized_neg1pos1_fold"
y_data_path = "/media/aisec-102/DATA3/Bibo2/Project/CT2MRI/dataset/CV/fold"
y_test_data_path = "/media/aisec-102/DATA3/Bibo2/Project/CT2MRI/dataset/CV/fold5.npz"
x_test_data_path = "/media/aisec-102/DATA3/rachel/data/CV/normalized_neg1pos1_fold5.npz"



# Load the data
for i in range(1, 5):
    x_path = f"{x_data_path}{i}.npz"
    y_path = f"{y_data_path}{i}.npz"
    print(x_path, y_path)
    if i == 1:
        x_data = np.load(x_path)
        y_data = np.load(y_path)
        x_train = x_data['x']
        y_train = (y_data['arr_1'] - 0.5) / 0.5
    else:
        x_data = np.load(x_path)
        y_data = np.load(y_path)
        x_train = np.concatenate((x_train, x_data['x']), axis=0)
        y_train = np.concatenate((y_train, (y_data['arr_1'] - 0.5) / 0.5), axis=0)

x_data_test = np.load(x_test_data_path)
x_test = x_data_test['x']
y_data_test = np.load(y_test_data_path)
y_test = (y_data_test['arr_1'] - 0.5) / 0.5

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Hyperparameters
batch_size = 32
num_epochs = 10  # Just for the sake of demonstration
total_timesteps = 10
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 1e-2
img_size = 256
img_channels = 1
first_conv_channels = 32
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2


# Build the unet model
model = uvit.UNetViTModel(
    timesteps=total_timesteps,
    learning_rate=learning_rate,
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
)

# Compile the model
model.compile(optimizer=model.optimizer, loss=model.loss)

# Train the model
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=1,
    batch_size=batch_size,
    callbacks=[modules.GanMonitor(x_test, y_test)],
)

