import os
from matplotlib import pyplot
import numpy as np

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2, name, dataname, step):
    images = np.vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Original']
    # scale from [-1,1] to [0,1]
    
    
    images = (images + 1) / 2.0
    # plot images row by row
    f = pyplot.figure(figsize=(12,4))
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(np.squeeze(images[i]),  cmap='gray')
        # title
        pyplot.title(titles[i])
    #
   
    sample_dir_ = 'generated_aligned/'+name+'/'+dataname+'/'
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    filename1 = sample_dir_+'%s_plot_%04d.png' % (name, (step+1))
    pyplot.savefig(filename1)
    #pyplot.show()
    pyplot.close()

    
# plot the image, the translation, and the reconstruction
def show_plot_generated(image, name = None, dataname = None, step = None):
    #images = np.vstack((imagesX, imagesY1, imagesY2))
    #titles = ['Real', 'Generated', 'Original']
    # scale from [-1,1] to [0,1]
    image = (image + 1) / 2.0
    f = pyplot.figure(figsize=(8,8))
    pyplot.axis('off')
    pyplot.imshow(np.squeeze(np.squeeze(image)),  cmap='gray')
    sample_dir_ = 'generated_test/'+name+'/'+dataname+'/'
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    filename1 = sample_dir_+'%s_plot_%04d.png' % (name, (step+1))
    pyplot.savefig(filename1)
    pyplot.show()
    pyplot.close()




import matplotlib.pyplot as plt
def show_plot_metrixs(stat, file_path, fid, xnumber=400, fid_new = 0):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    #plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,4))
    plt.plot(stat[:,0], color='blue', label='MSE = {:.5f} (SD: {:.5f})'.format(stat[:,0][:xnumber].mean(), stat[:,0][:xnumber].std()))        # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    #plt.ylim(0, 0.020)
    plt.ylabel("MSE")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path + "/mse.png")
    plt.close()

    fig = plt.figure(figsize=(10,4))
    plt.plot(stat[:,1], color='blue', label='MAE = {:.5f} (SD: {:.5f})'.format(stat[:,1][:xnumber].mean(), stat[:,1][:xnumber].std()))        # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    #plt.ylim(0.04, 0.10)
    plt.ylabel("MAE")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path +"/mae.png")
    plt.close()


    fig = plt.figure(figsize=(10,4))
    b = [ds[0][0] for ds in stat[:,2]]
    plt.plot(b, color='blue', label='Cosine Similarity = {:.5f} (SD: {:.5f})'.format(np.array(b)[:xnumber].mean(), np.array(b)[:xnumber].std()))        # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    #plt.ylim(0.93, 1);
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path +"/cs.png")
    plt.close()

    fig = plt.figure(figsize=(10,4))
    plt.plot(stat[:,3], color='blue', label='PSNR = {:.5f} (SD: {:.5f})'.format(stat[:,3][:xnumber].mean(), stat[:,3][:xnumber].std()))        # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    # plt.ylim(64, 74);
    plt.ylabel("PSNR")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path +"/psnr.png")
    plt.close()

    fig = plt.figure(figsize=(10,4))
    plt.plot(stat[:,4], color='blue', label='MAPE = {:.5f} (SD: {:.5f})'.format(stat[:,4][:xnumber].mean(), stat[:,4][:xnumber].std()))        # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    # plt.ylim(64, 74);
    plt.ylabel("MAPE")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path +"/mape.png")
    plt.close()


    fig = plt.figure(figsize=(10,4))
    plt.plot(stat[:,5], color='blue', label='Structural SIMilarity = {:.5f} (SD: {:.5f}) FID = {:.5f} '.format(stat[:,5][:xnumber].mean(), stat[:,5][:xnumber].std(), fid))    # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    # plt.ylim(64, 74);
    plt.ylabel("SSIM")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path +"/ssim.png")
    plt.close()

    fig = plt.figure(figsize=(10,4))
    plt.plot(stat[:,6], color='blue', label='Perceptual Loss = {:.5f} (SD: {:.5f}) FID new = {:.5f} '.format(stat[:,6][:xnumber].mean(), stat[:,6][:xnumber].std(), fid_new))    # specify color by name
    #mse, mae, cs, psnr
    plt.xlim(0, xnumber)
    # plt.ylim(64, 74);
    plt.ylabel("Perceptual Loss")
    plt.xlabel("Samples")
    #plt.axis('equal')
    plt.legend()
    plt.savefig(file_path +"/perloss.png")
    plt.close()




