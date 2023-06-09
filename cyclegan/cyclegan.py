
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
from . import modules
import loss
import evaluate
import os
import pandas as pd
from  matplotlib import pyplot as plt
from datetime import datetime

# CycleGAN
class CycleGAN(kr.Model):
    def __init__(self, flags,**kwargs):

        super().__init__(**kwargs)
        self.flags = flags

        # pass new name each time
        self.experiment_name = self.flags.name
        self.samples_dir = self.flags.sample_dir
        self.hist_path = self.flags.hist_path
        self.models_dir = self.flags.checkpoints_dir
        self.image_shape = (self.flags.crop_size, self.flags.crop_size, 1)
        self.image_size = self.flags.crop_size
        self.batch_size = self.flags.batch_size

        self.ssim_loss_coeff = self.flags.ssim_loss_coeff
        self.mae_loss_coeff = 0.5 * self.flags.mae_loss_coeff
        self.vgg_loss_coeff = self.flags.vgg_feature_loss_coeff
        self.disc_loss_coeff = self.flags.disc_loss_coeff
        self.cycle_loss_coeff = self.flags.cycle_loss_coeff
        self.identity_loss_coeff = self.flags.identity_loss_coeff



        self.discriminator_mri = modules.Discriminator(flags)
        self.discriminator_ct = modules.Discriminator(flags)
        self.generator_mri = modules.cyclegan_generator(flags)
        self.generator_ct = modules.cyclegan_generator(flags)
        self.combined_model = self.build_combined_model()

        # pass to flags: gen_lr 0.0002, gen_beta_1 0.5
        self.mri_g_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)
        self.ct_g_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)

        self.ct_discriminator_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
        self.mri_discriminator_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)

        self.discriminator_loss = loss.DiscriminatorLoss()
        self.vgg_loss = loss.VGGFeatureMatchingLoss()

    
        # MRI trackers.
        self.disc_loss_mri_tracker  = tf.keras.metrics.Mean(name="D_MRI")
        self.ssim_loss_mri_tracker= tf.keras.metrics.Mean(name="SSIM_MRI")
        self.mae_loss_mri_tracker= tf.keras.metrics.Mean(name="MAE_MRI")
        self.vgg_loss_mri_tracker = tf.keras.metrics.Mean(name="VGG_MRI")
        self.cycle_loss_mri_tracker= tf.keras.metrics.Mean(name="Cy_MRI")
        self.identity_loss_mri_tracker= tf.keras.metrics.Mean(name="Id_MRI")
        #CT trackers
        self.disc_loss_ct_tracker= tf.keras.metrics.Mean(name="D_CT")
        self.ssim_loss_ct_tracker= tf.keras.metrics.Mean(name="SSIM_CT")
        self.mae_loss_ct_tracker= tf.keras.metrics.Mean(name="MAE_CT")
        self.vgg_loss_ct_tracker = tf.keras.metrics.Mean(name="VGG_CT")
        self.cycle_loss_ct_tracker= tf.keras.metrics.Mean(name="Cy_CT")
        self.identity_loss_ct_tracker= tf.keras.metrics.Mean(name="Id_CT")


    @property
    def metrics(self):
        return [self.disc_loss_mri_tracker,
            self.ssim_loss_mri_tracker,
            self.mae_loss_mri_tracker,
            self.vgg_loss_mri_tracker,
            self.cycle_loss_mri_tracker,
            self.identity_loss_mri_tracker,
            self.disc_loss_ct_tracker,
            self.ssim_loss_ct_tracker,
            self.mae_loss_ct_tracker,
            self.vgg_loss_ct_tracker,
            self.cycle_loss_ct_tracker,
            self.identity_loss_ct_tracker]


    def build_combined_model(self):

        self.discriminator_mri.trainable = False
        self.discriminator_ct.trainable = False
        
        ct_input = kr.Input(shape=self.image_shape, name="ct")
        mri_input = kr.Input(shape=self.image_shape, name="mri")

        generated_mri = self.generator_mri(ct_input)
        generated_ct = self.generator_ct(mri_input)
        id_mri = self.generator_mri(mri_input)
        id_ct = self.generator_ct(ct_input)
        cycled_mri = self.generator_mri(generated_ct)
        cycled_ct = self.generator_ct(generated_mri)

        
       
        combined_model = kr.Model(
            [ct_input, mri_input],
            [generated_mri, generated_ct,
             id_mri, id_ct,
             cycled_mri, cycled_ct
             ])

        return combined_model


    def compile(self, **kwargs):
        super().compile(**kwargs)


    def train_discriminator(self, real_ct, real_mri):
        
        fake_mri = self.generator_mri(real_ct)
        fake_ct = self.generator_ct(real_mri)

        with tf.GradientTape(persistent=True) as gradient_tape:
            pred_fake_mri = self.discriminator_mri([real_mri, fake_mri])  
            pred_real_mri = self.discriminator_mri([real_mri, real_mri])  
            loss_fake_mri = self.discriminator_loss(False, pred_fake_mri)
            loss_real_mri = self.discriminator_loss(True, pred_real_mri)
            total_loss_mri = self.disc_loss_coeff * (loss_fake_mri + loss_real_mri)
        
        with tf.GradientTape(persistent=True) as ct_gradient_tape:
            pred_fake_ct = self.discriminator_ct([real_ct, fake_ct]) 
            pred_real_ct = self.discriminator_ct([real_ct, real_ct])
            loss_fake_ct = self.discriminator_loss(False, pred_fake_ct)
            loss_real_ct = self.discriminator_loss(True, pred_real_ct)
            total_loss_ct = self.disc_loss_coeff * (loss_fake_ct + loss_real_ct)

        #total_loss = total_loss_ct + total_loss_mri


        #dis_trainable  =(
            #self.discriminator_mri.trainable_variables +
            #self.discriminator_ct.trainable_variables 
            #)
        
        gradients_mri = gradient_tape.gradient(
            total_loss_mri, self.discriminator_mri.trainable_variables) 

        self.mri_discriminator_optimizer.apply_gradients(
            zip(gradients_mri, self.discriminator_ct.trainable_variables)
        )



        gradients_ct = ct_gradient_tape.gradient(
            total_loss_ct, self.discriminator_ct.trainable_variables) 

        self.ct_discriminator_optimizer.apply_gradients(
            zip(gradients_ct, self.discriminator_ct.trainable_variables)
        )

        

        return total_loss_mri, total_loss_ct


    def train_generator(self, ct, mri__):

        self.discriminator_mri.trainable = False
        self.discriminator_ct.trainable = False

        with tf.GradientTape(persistent=True) as mri_tape:
            generated_mri, generated_ct, id_mri, id_ct, cycled_mri, cycled_ct = self.combined_model([ct, mri__])
    
            ssim_loss_mri = self.ssim_loss_coeff * loss.SSIMLoss(mri__, generated_mri)
            mae_loss_mri = self.mae_loss_coeff * loss.mae(mri__, generated_mri)
            vgg_loss_mri = self.vgg_loss_coeff * self.vgg_loss(mri__, generated_mri)
            cycle_loss_mri = self.cycle_loss_coeff * loss.mae(mri__, cycled_mri)
            cycle_loss_ct = self.cycle_loss_coeff * loss.mae(ct, cycled_ct)
            identity_loss_mri = self.identity_loss_coeff * loss.mae(mri__, id_mri)
            total_loss_mri = ssim_loss_mri + mae_loss_mri + vgg_loss_mri + cycle_loss_mri + cycle_loss_ct + identity_loss_mri

        with tf.GradientTape(persistent=True) as ct_tape:
            generated_mri, generated_ct, id_mri, id_ct, cycled_mri, cycled_ct = self.combined_model([ct, mri__])

            ssim_loss_ct = self.ssim_loss_coeff * loss.SSIMLoss(ct, generated_ct)
            mae_loss_ct = self.mae_loss_coeff * loss.mae(ct, generated_ct)
            vgg_loss_ct = self.vgg_loss_coeff * self.vgg_loss(ct, generated_ct)
            cycle_loss_mri = self.cycle_loss_coeff * loss.mae(mri__, cycled_mri)
            cycle_loss_ct = self.cycle_loss_coeff * loss.mae(ct, cycled_ct)
            identity_loss_ct = self.identity_loss_coeff * loss.mae(ct, id_ct)
            total_loss_ct = ssim_loss_ct + mae_loss_ct + vgg_loss_ct + cycle_loss_ct + cycle_loss_mri + identity_loss_ct

        

        mri_g_variables = self.generator_mri.trainable_variables
        mri_g_gradients = mri_tape.gradient(total_loss_mri, mri_g_variables)
        self.mri_g_optimizer.apply_gradients(zip(mri_g_gradients , mri_g_variables))

        ct_g_variables = self.generator_ct.trainable_variables
        ct_g_gradients = ct_tape.gradient(total_loss_ct, ct_g_variables)
        self.ct_g_optimizer.apply_gradients(zip(ct_g_gradients , ct_g_variables))

            
        
        #g_trainable  = (self.combined_model.trainable_variables)
        #mri_g_variables = (self.generator_mri.trainable_variables)
        #g_gradients = tape.gradient(total_loss, self.combined_model.trainable_variables)
        #self.mri_g_optimizer.apply_gradients(zip(g_gradients , self.combined_model.trainable_variables))


        losses__ = (ssim_loss_mri, mae_loss_mri, vgg_loss_mri, cycle_loss_mri, identity_loss_mri,
                    ssim_loss_ct, mae_loss_ct, vgg_loss_ct, cycle_loss_ct, identity_loss_ct)

        return losses__ 


    def train_step(self, data):

        ct, mri = data

        total_loss_mri, total_loss_ct = self.train_discriminator(ct, mri)
        losses = self.train_generator(ct, mri)
        (ssim_loss_mri, mae_loss_mri, vgg_loss_mri, cycle_loss_mri, identity_loss_mri, ssim_loss_ct, mae_loss_ct, vgg_loss_ct, cycle_loss_ct, identity_loss_ct) = losses

        # MRI trackers.
        self.disc_loss_mri_tracker.update_state(total_loss_mri)
        self.ssim_loss_mri_tracker.update_state(ssim_loss_mri)
        self.mae_loss_mri_tracker.update_state(mae_loss_mri)
        self.vgg_loss_mri_tracker.update_state(vgg_loss_mri)
        self.cycle_loss_mri_tracker.update_state(cycle_loss_mri)
        self.identity_loss_mri_tracker.update_state(identity_loss_mri)
        #CT trackers
        self.disc_loss_ct_tracker.update_state(total_loss_ct)
        self.ssim_loss_ct_tracker.update_state(ssim_loss_ct)
        self.mae_loss_ct_tracker.update_state(mae_loss_ct)
        self.vgg_loss_ct_tracker.update_state(vgg_loss_ct)
        self.cycle_loss_ct_tracker.update_state(cycle_loss_ct)
        self.identity_loss_ct_tracker.update_state(identity_loss_ct)


        results = {m.name: m.result() for m in self.metrics}

        return results


    def test_step(self, data):

        ct, mri = data
        fake_mri = self.generator_mri(ct)
        fake_ct = self.generator_ct(mri)
        pred_fake_mri = self.discriminator_mri([mri, fake_mri]) 
        pred_real_mri = self.discriminator_mri([mri, mri]) 
        loss_fake_mri = self.discriminator_loss(False, pred_fake_mri)
        loss_real_mri = self.discriminator_loss(True, pred_real_mri)


        pred_fake_ct = self.discriminator_ct([ct, fake_ct])
        pred_real_ct = self.discriminator_ct([ct, ct])
        loss_fake_ct = self.discriminator_loss(False, pred_fake_ct)
        loss_real_ct = self.discriminator_loss(True, pred_real_ct)


        total_loss_mri = self.disc_loss_coeff * (loss_fake_mri + loss_real_mri)
        total_loss_ct = self.disc_loss_coeff * (loss_fake_ct + loss_real_ct)


        generated_mri, generated_ct, id_mri, id_ct, cycled_mri, cycled_ct = self.combined_model([ct, mri])
        ssim_loss_mri = self.ssim_loss_coeff * loss.SSIMLoss(mri, generated_mri)
        mae_loss_mri = self.mae_loss_coeff * loss.mae(mri, generated_mri)
        vgg_loss_mri = self.vgg_loss_coeff * self.vgg_loss(mri, generated_mri)
        ssim_loss_ct = self.ssim_loss_coeff * loss.SSIMLoss(ct, generated_ct)
        mae_loss_ct = self.mae_loss_coeff * loss.mae(ct, generated_ct)
        vgg_loss_ct = self.vgg_loss_coeff * self.vgg_loss(ct, generated_ct)
        cycle_loss_mri = self.cycle_loss_coeff * loss.mae(mri, cycled_mri)
        cycle_loss_ct = self.cycle_loss_coeff * loss.mae(ct, cycled_ct)
        identity_loss_mri = self.identity_loss_coeff * loss.mae(mri, id_mri)
        identity_loss_ct = self.identity_loss_coeff * loss.mae(ct, id_ct)

        # MRI trackers.
        self.disc_loss_mri_tracker.update_state(total_loss_mri)
        self.ssim_loss_mri_tracker.update_state(ssim_loss_mri)
        self.mae_loss_mri_tracker.update_state(mae_loss_mri)
        self.vgg_loss_mri_tracker.update_state(vgg_loss_mri)
        self.cycle_loss_mri_tracker.update_state(cycle_loss_mri)
        self.identity_loss_mri_tracker.update_state(identity_loss_mri)
        #CT trackers
        self.disc_loss_ct_tracker.update_state(total_loss_ct)
        self.ssim_loss_ct_tracker.update_state(ssim_loss_ct)
        self.mae_loss_ct_tracker.update_state(mae_loss_ct)
        self.vgg_loss_ct_tracker.update_state(vgg_loss_ct)
        self.cycle_loss_ct_tracker.update_state(cycle_loss_ct)
        self.identity_loss_ct_tracker.update_state(identity_loss_ct)

        results = {m.name: m.result() for m in self.metrics}

        return results

    def call(self, inputs):
        return self.generator_mri(inputs)

    
    def model_evaluate(self, test_data, epoch=0):

        results = []

        num_batches = len(test_data[0]//self.batch_size)

        for i in range(0, num_batches, self.batch_size):
            ct, mri = test_data[0][i:i+self.batch_size], test_data[1][i:i+self.batch_size]

        #for ct, mri in test_data:
            
            fake_mri = self.generator_mri(ct)

            fid = evaluate.calculate_fid(mri, fake_mri, 
				input_shape=(self.flags.crop_size, self.flags.crop_size, 3))

            mse, mae, cs, psnr, ssim = evaluate.get_metrics(mri, fake_mri)

            results.append([fid, mse, mae, cs, psnr, ssim])
            print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))

        results = np.array(results, dtype=object).mean(axis=0)

        filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        results_dir = os.path.join(self.flags.result_logs, self.flags.name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        log_file = os.path.join(results_dir, filename)
        np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")


    def save_model(self, flags):
        # model_path = '/media/aisec1/DATA3/rachel/pcxgan/models/Pix2Pix_test'
        self.generator_mri.save(self.flags.model_path)


    def plot_losses(self, hist):
        
        exp_path = self.hist_path + self.experiment_name

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)


        # save history to csv   
        hist_df = pd.DataFrame(hist) 
        hist_df.to_csv(exp_path + '/cyclegan_hist.csv')

        losses = ["D_MRI", "SSIM_MRI", "MAE_MRI", "VGG_MRI", "Cy_MRI", "Id_MRI", "D_CT", "SSIM_CT", "MAE_CT", "VGG_CT", "Cy_CT", "Id_CT"]

        # plot losses
        for loss in losses:
            plt.figure()
            plt.plot(hist[loss])
            plt.plot(hist['val_' + loss])
            plt.legend([loss,'val_' + loss],loc='upper right')
            plt.savefig(exp_path + '/cyclegan_' + loss + '_loss.png')
