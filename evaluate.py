import os
import numpy as np
import tensorflow as tf
import tensorflow.io as tfio
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import data_helper
import plot
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
import math
import sklearn.metrics as sk
import data_loader
import modules


def normalize_scale(x, file=''):
  ok_=True
  if x.max() - x.min()!=0:
    x1 = (x-x.min())/(x.max()-x.min())
  else:
    print(file)
    ok_=False
    x1 = np.zeros(x.shape)
  x1 = (x1 - 0.5) / 0.5
  return  x1, ok_


def load_image(image_path):
  image_bytes = tf.io.read_file(image_path)
  image_ = tfio.image.decode_dicom_image(image_bytes,dtype=tf.uint16)
  image_, _  = normalize_scale(np.squeeze(image_.numpy()))
  image_ = np.expand_dims(image_, axis=2)
  return image_



def predict_image (model_file, image):
  
  model_AtoB = load_model(model_file)
  sample = load_image(image)
  
  A_sample = np.expand_dims(sample, axis=0)
  B_generated  = model_AtoB.predict(A_sample)
  
  modelname = model_file.split('/')[-1][:-3]
  print(modelname)
  
  data_name = image.split('/')[-1][:-4]
  plot.show_plot_generated(B_generated[0]*3500, modelname, data_name, 0 )


def predict( model, image_ct ):
  #image_ct = 'dataset/dataset/Benters, T 0266877/IMAGE-DataSet#1/CT0009.dcm'
  #'dataset/dataset_test/test-sets/IMAGE-DataSet#1/CT0009.dcm'
  image_ = load_image(image_ct)
  f = pyplot.figure(figsize=(8,8))
  pyplot.imshow(np.squeeze(image_), cmap='gray')
  
  #model = 'models/Pix2Pix/new/model_1466520.h5'
  predict_image(model, image_ct)




def read_files(path):
  CT_IMAGES  = []
  MRI_IMAGES = []
  for root, dirs, files in os.walk(path):
    for file in files:
      if file.endswith(".dcm") and file.startswith("CT"):
        MRI_FILE = "MRI"+file[2:]
        if os.path.exists(os.path.join(root, MRI_FILE)):
          CT_IMAGES.append(os.path.join(root, file))
          MRI_IMAGES.append(os.path.join(root, MRI_FILE))
  
  return CT_IMAGES, MRI_IMAGES

# load all images in a directory into memory
def load_predict (path, model_path):
  
  model = load_model(model_path)
  
  dir_name = '/'.join(path.split('/')[2:])
  print(dir_name)
  
  dest_dir = 'dataset/dataset2/'+dir_name+'/'
  print(dest_dir)
  
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  
  CT_IMAGES, MRI_IMAGES = read_files(path)
  
  for ct, mri in zip(CT_IMAGES, MRI_IMAGES):
    filename_ct = (ct.split('/')[-1])[:-4]+'.png'
    filename_mri =( mri.split('/')[-1])[:-4]+'.png'
    
    image_bytes_CT = tf.io.read_file(ct)
    image_bytes_MRI = tf.io.read_file(mri)
    
    #print(image_bytes_CT.shape)
    
    image_CT = tfio.image.decode_dicom_image(image_bytes_CT,dtype=tf.uint16)
    image_MRI = tfio.image.decode_dicom_image(image_bytes_MRI,dtype=tf.uint16)
    
    image_CT  = np.squeeze(image_CT.numpy())
    image_MRI = np.squeeze(image_MRI.numpy())
    
    
    pyplot.image.imsave(dest_dir+filename_ct, image_CT, cmap=pyplot.cm.gray)
    pyplot.image.imsave(dest_dir+filename_mri, image_MRI, cmap=pyplot.cm.gray)
    
    image_CT, _  = data_helper.normalize_minmax(image_CT, ct)
    image_CT = np.expand_dims(image_CT, axis=2)
    #print(image_CT.shape)
    
    image_CT = np.expand_dims(image_CT, axis=0)
    
    generated_mri = model.predict(image_CT)
    
    #generated_mri = (np.squeeze(generated_mri[0]) + 1) / 2.0
    generated_mri = generated_mri[0]*3500
    
    generated_mri = (generated_mri + 1) / 2.0
    
    pyplot.image.imsave(dest_dir+"G-"+filename_mri, np.squeeze(generated_mri), cmap=pyplot.cm.gray)



# calculate frechet inception distance
def calculate_fid(y_true, y_pred, input_shape = (512,512,3)):
  
  model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
  
  # convert integer to floating point values
  # y_true = y_true.astype('float32')
  # y_pred = y_pred.astype('float32')
  
  y_true = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_true))
  y_pred = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_pred))
  
  # pre-process images
  y_true = preprocess_input(y_true)
  y_pred = preprocess_input(y_pred)
  
  y_true = model.predict(y_true)
  y_pred = model.predict(y_pred)
  # calculate mean and covariance statistics
  mu1, sigma1 = y_true.mean(axis=0), np.cov(y_true, rowvar=False)
  mu2, sigma2 = y_pred.mean(axis=0), np.cov(y_pred, rowvar=False)
  
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = sqrtm(np.matrix(sigma1).dot(np.matrix(sigma2)))
  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid


# get MSE, MAE, CS, PSNR, SSIM
def get_metrics___(y_true_, y_pred_):
  
  mse, mae, cs, psnr,ssim =[],[],[],[],[]
  
  for y_true,y_pred in zip(y_true_ ,y_pred_):
    
    y_true,y_pred = y_true.numpy(), y_pred.numpy()
    mse.append(sk.mean_squared_error(y_true.flatten(),y_pred.flatten()))
    mae.append(sk.mean_absolute_error(y_true.flatten(),y_pred.flatten()))
    cs.append(sk.pairwise.cosine_similarity([y_true.flatten()], [y_pred.flatten()]))
    
    if mse[-1] == 0:
      psnr_ =  float('inf')
      psnr.append(psnr_)
    else:
      psnr.append(20 * math.log10(1 / math.sqrt(mse[-1])))
    #fid = 0 #calculate_fid(x,y)
    
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0
    
    im1 = tf.image.convert_image_dtype(y_true, tf.float32)
    im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
    ssim.append(float(tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                                    filter_sigma=1.5, k1=0.01, k2=0.03)[0]))
    #print("ssim:", ssim)
  
  
  
  
  return np.array(mse).mean(), np.array(mae).mean(), np.array(cs).mean(), np.array(psnr).mean(), np.array(ssim).mean()




def get_metrics(y_true, y_pred):
  
  
  y_true,y_pred = y_true.numpy(), y_pred.numpy()
  mse=sk.mean_squared_error(y_true.flatten(),y_pred.flatten())
  mae=sk.mean_absolute_error(y_true.flatten(),y_pred.flatten())
  cs=sk.pairwise.cosine_similarity([y_true.flatten()], [y_pred.flatten()])
  
  if mse == 0:
    psnr =  float('inf')
  
  else:
    psnr=20 * math.log10(1 / math.sqrt(mse))
  #fid = 0 #calculate_fid(x,y)
  
  y_true = (y_true + 1.0) / 2.0
  y_pred = (y_pred + 1.0) / 2.0
  
  im1 = tf.image.convert_image_dtype(y_true, tf.float32)
  im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
  ssim=float(tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                           filter_sigma=1.5, k1=0.01, k2=0.03)[0])
  #print("ssim:", ssim)
  
  
  
  
  return mse, mae, cs, psnr,ssim




def predict_pcx(flags, decoder_file, label):
  
  #encoder = load_model(encoder_file)
  decoder = load_model(decoder_file)
  latent_vector = tf.random.normal(
      shape=(flags.batch_size, flags.latent_dim), mean=0.0, stddev=2.0)
  
  
  #print(samples.shape)
  generated = decoder([tf.convert_to_tensor(latent_vector), tf.convert_to_tensor(label)])
  
  modelname = 'PCxGAN_fold1245'
  print(modelname)
  
  data_name = 'test'
  
  counter = 0
  for image in generated:
    plot.show_plot_generated(image, modelname, data_name + str(counter), 0)
    counter += 1
  
  return generated


def predict_p2p(model_path, ct):
  

  generator = load_model(model_path)
  
  generated = generator(ct)
  
  modelname = 'Pix2Pix_fold2345'
  print(modelname)
  
  data_name = 'test'
  
  counter = 0
  for image in generated:
    plot.show_plot_generated(image, modelname, data_name + str(counter), 0)
    counter += 1
  
  return generated





if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  print(len(gpus))
  if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
      tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
      # Visible devices must be set at program startup
      print(e)
  
  
  
  BATCH_SIZE = 4
  datafile = '/media/aisec1/DATA3/Bibo2/Project/CT2MRI/dataset/CT_MRI-512-Updated.npz'
  test_dataset = data_loader.DataGenerator(datafile, batch_size=BATCH_SIZE, is_train=False, remove_bad_images=True).load()
  
  encoder_path = '/media/aisec1/DATA3/rachel/PCGAN/models/PCxGAN_e'
  decoder_path = '/media/aisec1/DATA3/rachel/PCGAN/models/PCxGAN_d'
  sampler_path = '/media/aisec1/DATA3/rachel/PCGAN/models/PCxGAN_s'
  
  for ct, mri, label in test_dataset:
    predict_test(encoder_path, decoder_path, sampler_path, ct, label)
  
  '''
  y = np.concatenate([y for x, y, mask in test_dataset], axis=0)
  x = np.concatenate([x for x, y, mask in test_dataset], axis=0)

  # predict on test set
  y_pred = predict_test(encoder_path, decoder_path, x)

  # get evaluation metrics
  fid = calculate_fid(y, y_pred)
  mse,mae,cs,psnr,ssim = get_metrics(y, y_pred)
  print(fid)
  print(mse)
  print(mae)
  print(cs)
  print(psnr)
  print(ssim)
  '''
