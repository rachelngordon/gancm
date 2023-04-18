from pcxgan.pcxgan_ct import PCxGAN_ct
from flags import Flags
import data_loader
import pcxgan.modules_ct as modules

flags = Flags().parse()


train_dataset = data_loader.DataGenerator_Ready(flags, flags.data_path).load()
test_dataset = data_loader.DataGenerator_Ready(flags, flags.test_data_path).load()


#Build and train the model
model = PCxGAN_ct(flags)
model.compile()
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=flags.epochs,
  verbose=1,
  callbacks=[modules.GanMonitor(test_dataset, flags)],
)


#model.plot_losses(history.history)
#model.model_evaluate(test_dataset)
model.save_model(flags)

