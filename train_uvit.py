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
import time
import data_loader
from flags import Flags



def main(flags):

    # pass path to data in flags

    train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
    test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


    # Start the timer
    start_time = time.time()

    # Hyperparameters
    batch_size = 32
    num_epochs = 2 # Just for the sake of demonstration
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
        train_data,
        validation_data=test_data,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[modules.GanMonitor(test_data)],
    )


    end_time = time.time()

    # Calculate the training duration
    training_duration = end_time - start_time

    # Print the training time
    print("Training time: {:.2f} seconds".format(training_duration))

    # Save the training time to a file
    #filename = '/grand/EVITA/ct-mri/new_edge_results/time_train/' + flags.exp_name + "_train_time.txt"
    #   with open(filename, "w") as file:
    #       file.write("Training time: {:.2f} seconds".format(training_duration))
    #       print("Training time saved to", filename)


    model.save_model('/media/aisec-102/DATA3/rachel/experiments/models/uvit_test')
    #model.model_evaluate(test_data)
    #model.plot_losses(history.history)


if __name__ == '__main__':
    flags = Flags().parse()
    main(flags)