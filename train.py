from p2p.pix2pix import pix2pix
from flags import Flags
import data_loader
import p2p.modules as modules

flags = Flags().parse()


train_dataset = data_loader.DataGenerator_PairedReady(flags, flags.data_path).load()
test_dataset = data_loader.DataGenerator_PairedReady(flags, flags.test_data_path).load()


#Build and train the model
model = pix2pix(flags)
model.compile()
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=flags.epochs,
  verbose=1,
  callbacks=[modules.P2PMonitor(test_dataset, flags)],
)


model.plot_losses(history.history)
model.model_evaluate(test_dataset)
model.save_model(flags)

