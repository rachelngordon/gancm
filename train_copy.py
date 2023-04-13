import sys
sys.path.append('/media/aisec-102/DATA3/rachel/pcxgan/cyclegan')

from cyclegan import cyclegan
from flags import Flags
import data_loader
import modules
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

flags = Flags().parse()


train_dataset = data_loader.DataGenerator_PairedReady(flags, flags.data_path).load()
test_dataset = data_loader.DataGenerator_PairedReady(flags, flags.test_data_path).load()


#Build and train the model
model = cyclegan(flags)
model.compile()
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=flags.epochs,
  verbose=1,
  callbacks=[modules.CycleMonitor(test_dataset, flags)],
)


model.plot_losses(history.history)
model.model_evaluate(test_dataset)
model.save_model()

