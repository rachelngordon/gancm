import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
import math
import sklearn.metrics as sk
from datetime import datetime


# calculate frechet inception distance
def calculate_fid(y_true, y_pred, input_shape = (256,256,3)):
  
  model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
  
  y_true = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_true))
  y_pred = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_pred))

  y_true = preprocess_input(y_true)
  y_pred = preprocess_input(y_pred)
  
  y_true = model.predict(y_true)
  y_pred = model.predict(y_pred)

  mu1, sigma1 = y_true.mean(axis=0), np.cov(y_true, rowvar=False)
  mu2, sigma2 = y_pred.mean(axis=0), np.cov(y_pred, rowvar=False)
  
  ssdiff = np.sum((mu1 - mu2)**2.0)

  covmean = sqrtm(np.matrix(sigma1).dot(np.matrix(sigma2)))

  if np.iscomplexobj(covmean):
    covmean = covmean.real

  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid


def get_metrics(y_true, y_pred):
  
  
  y_true,y_pred = np.array(y_true), np.array(y_pred)
  mse=sk.mean_squared_error(y_true.flatten(),y_pred.flatten())
  mae=sk.mean_absolute_error(y_true.flatten(),y_pred.flatten())
  cs=sk.pairwise.cosine_similarity([y_true.flatten()], [y_pred.flatten()])
  
  if mse == 0:
    psnr =  float('inf')
  
  else:
    psnr=20 * math.log10(1 / math.sqrt(mse))
  
  y_true = (y_true + 1.0) / 2.0
  y_pred = (y_pred + 1.0) / 2.0
  
  im1 = tf.image.convert_image_dtype(y_true, tf.float32)
  im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
  ssim=float(tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)[0])
  #ssim=float(tf.image.ssim(im1, im2, 1)[0])

  return mse, mae, cs, psnr,ssim


def pix2pix_evaluate(dir_, generator, test_data, epoch=0):


        results = []
        
        #num_batches = len(test_data[0]//self.batch_size)

        #for i in range(0, num_batches, self.batch_size):
            #ct, mri = test_data[0][i:i+self.batch_size], test_data[1][i:i+self.batch_size]


        for ct, mri in test_data:
            
            fake_mri = generator(ct)

            mri = (mri + 1.0) / 2.0
            fake_mri = (fake_mri + 1.0) / 2.0
            

            fid = calculate_fid(mri, fake_mri, 
				    input_shape=(256, 256, 3))

            mse, mae, cs, psnr, ssim = get_metrics(mri, fake_mri)

            results.append([fid, mse, mae, cs, psnr, ssim])
            print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))

        results = np.array(results, dtype=object).mean(axis=0)

        filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        results_dir = os.path.join('/media/aisec-102/DATA3/rachel/experiments/results/test_data/', dir_)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        log_file = os.path.join(results_dir, filename)
        print("File path: ", log_file)
        np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")





def pcxgan_evaluate(dir_, decoder, test_data, epoch=0):
  results = []
  
  #for ct, mri, label in test_data:
  for ct, mri, mask in test_data:

    # Sample latent from a normal distribution.
    latent_vector = tf.random.normal(
      shape=(1, 256), mean=0.0, stddev=2.0
    )
    fake_image = decoder([latent_vector, mask, ct])
    
    mri = (mri + 1.0) / 2.0
    fake_image = (fake_image + 1.0) / 2.0
    
    fid = calculate_fid(mri, fake_image,
                                  input_shape=(
                                    256,
                                    256, 3))
    mse, mae, cs, psnr, ssim = get_metrics(mri, fake_image)
    
    results.append([fid, mse, mae, cs, psnr, ssim])
    print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))
  

  results = np.array(results, dtype=object).mean(axis=0)
  
  filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
  results_dir = os.path.join('/media/aisec-102/DATA3/rachel/experiments/results/test_data/', dir_)
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  log_file = os.path.join(results_dir, filename)
  print("File path: ", log_file)
  np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")


class DataGenerator_Ready(kr.utils.Sequence):
    def __init__(self, data_path, if_train = True, **kwargs):
        
        super().__init__(**kwargs)

        
        self.data_path = data_path
        self.batch_size = 1
        x, y, z = self.load_data(self.data_path, if_train=if_train)
        self.dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
        self.dataset.shuffle(buffer_size=10, seed=42, reshuffle_each_iteration=False)
        self.dataset = self.dataset.map(
        lambda x, y, z: (x, y, tf.one_hot(tf.squeeze(tf.cast(z, tf.int32)), 2)), num_parallel_calls=tf.data.AUTOTUNE)

        
        
        
    def load_data(self, data_path, if_train=True):
            
        if if_train:
        
            for i in [1,2,3,4]:
                path = f"{data_path}{i}.npz"
                if i == 1:
                    data = np.load(path)
                    x, y, z = data['arr_0'], data['arr_1'], data['arr_2']
                else:
                    data = np.load(path)
                    x = np.concatenate((x, data['arr_0']), axis=0)
                    y = np.oncatenate((y, data['arr_1']), axis=0)
                    z = np.oncatenate((z, data['arr_2']), axis=0)

            
            return x, y, z

        else: 
            path = f"{data_path}.npz"
            data = np.load(path)
            x, y, z = data['arr_0'], data['arr_1'], data['arr_2']
            return x, y, z
        
    def __getitem__(self, idx):
        return self.dataset.batch(self.batch_size, drop_remainder=True)
    
    def load(self):
        return self.dataset.batch(self.batch_size, drop_remainder=True)
	

test_dataset = DataGenerator_Ready('/media/aisec-102/DATA3/rachel/data/test_data/avg_eq_seg_test', if_train=False).load()

# p2p_path = "/media/aisec-102/DATA3/rachel/experiments/models/p2p/p2p_avg_eq_1234"
# p2p = load_model(p2p_path)
# p2p.compile()
# cy_path = "/media/aisec-102/DATA3/rachel/experiments/models/cyclegan/cy_avg_eq_1234/g_mri"
# cy = load_model(cy_path)
# cy.compile()
# unet_path = "/media/aisec-102/DATA3/rachel/experiments/models/unet/unet_avg_eq_1234"
# unet = load_model(unet_path)
# unet.compile()
gancm_avg_eq_path = "/media/aisec-102/DATA3/rachel/experiments/models/pcx_eq_new/pcx_seg_avg_eq_1234_d"
gancm_avg_eq = load_model(gancm_avg_eq_path)
gancm_avg_eq.compile()

gancm_no_eq_path = "/media/aisec-102/DATA3/rachel/experiments/models/pcx_eq_new/pcx_seg_no_eq_1234_d"
gancm_no_eq = load_model(gancm_no_eq_path)
gancm_no_eq.compile()

gancm_eq_path = "/media/aisec-102/DATA3/rachel/experiments/models/pcx_eq_new/pcx_seg_eq_1234_d"
gancm_eq = load_model(gancm_eq_path)
gancm_eq.compile()

pcxgan_evaluate('gancm/ct/avg_eq', gancm_avg_eq, test_dataset)
print("GANCM Avg Eq Complete.")

pcxgan_evaluate('gancm/ct/no_eq', gancm_no_eq, test_dataset)
print("GANCM No Eq Complete.")

pcxgan_evaluate('gancm/ct/eq', gancm_eq, test_dataset)
print("GANCM Eq Complete.")
