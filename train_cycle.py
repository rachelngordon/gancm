# import sys
# sys.path.append('/media/aisec-102/DATA3/rachel/pcxgan/cyclegan')

from cyclegan.cyclegan import cyclegan
from flags import Flags
import data_loader
import cyclegan.modules as modules
import os


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
  callbacks=[modules.CycleMonitor(test_dataset, flags)],
)


model.plot_losses(history.history)
model.model_evaluate(test_dataset)
model.save_model(flags)

