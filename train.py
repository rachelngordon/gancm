import sys
sys.path.append('/media/aisec-102/DATA3/rachel/pcxgan/p2p')


from pix2pix import pix2pix
from flags import Flags
import data_loader
import modules
import pandas as pd
from  matplotlib import pyplot as plt

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

# save history to csv   
hist_df = pd.DataFrame(history.history) 
hist_df.to_csv('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/pix2pix_hist.csv')

# plot discriminator loss
plt.figure()
plt.plot(history.history['disc_loss'])
plt.plot(history.history['val_disc_loss'])
plt.legend(['disc_loss','val_disc_loss'],loc='upper right')
plt.savefig('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/p2p_disc_loss.png')

# plot generator loss
plt.figure()
plt.plot(history.history['gen_loss'])
plt.plot(history.history['val_gen_loss'])
plt.legend(['gen_loss','val_gen_loss'],loc='upper right')
plt.savefig('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/p2p_gen_loss.png')

# plot feature loss
plt.figure()
plt.plot(history.history['feat_loss'])
plt.plot(history.history['val_feat_loss'])
plt.legend(['feat_loss','val_feat_loss'],loc='upper right')
plt.savefig('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/p2p_feat_loss.png')

# plot vgg loss
plt.figure()
plt.plot(history.history['vgg_loss'])
plt.plot(history.history['val_vgg_loss'])
plt.legend(['vgg_loss','val_vgg_loss'],loc='upper right')
plt.savefig('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/p2p_vgg_loss.png')

# plot ssim loss
plt.figure()
plt.plot(history.history['ssim_loss'])
plt.plot(history.history['val_ssim_loss'])
plt.legend(['ssim_loss','val_ssim_loss'],loc='upper right')
plt.savefig('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/p2p_ssim_loss.png')

# plot mae loss
plt.figure()
plt.plot(history.history['mae_loss'])
plt.plot(history.history['val_mae_loss'])
plt.legend(['mae_loss','val_mae_loss'],loc='upper right')
plt.savefig('/media/aisec-102/DATA3/rachel/pcxgan/p2p/history/p2p_mae_loss.png')


#model.model_evaluate(test_dataset)
#model.save_model()

