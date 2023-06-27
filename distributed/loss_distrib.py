import tensorflow as tf
import tensorflow.keras as kr
import numpy as np

def SSIMLoss(y_true, y_pred, strategy_obj):
    with strategy_obj.scope():
        y_true = (y_true + 1.0) / 2.0
        y_pred = (y_pred + 1.0) / 2.0
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0), reduction=tf.keras.losses.Reduction.SUM)

def ssim_l2_a_loss(y_true, y_pred, strategy_obj):
    with strategy_obj.scope():
        yy_true = (y_true + 1.0) / 2.0
        yy_pred = (y_pred + 1.0) / 2.0
        return (1 - tf.reduce_mean(tf.image.ssim(yy_true, yy_pred, 1.0), reduction=tf.keras.losses.Reduction.SUM)) + 0.00005*tf.reduce_sum(tf.math.squared_difference(yy_true,yy_pred), reduction=tf.keras.losses.Reduction.SUM)

def l1_l2_loss(y_true, y_pred, strategy_obj):
    with strategy_obj.scope():
        return tf.reduce_mean(tf.abs(y_true - y_pred), reduction=tf.keras.losses.Reduction.SUM) + tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), reduction=tf.keras.losses.Reduction.SUM)


def mae(y_true, y_pred, strategy_obj):
    with strategy_obj.scope():
        y_true = (y_true + 1.0) / 2.0
        y_pred = (y_pred + 1.0) / 2.0
        return tf.reduce_mean(tf.abs(y_true - y_pred), reduction=tf.keras.losses.Reduction.SUM) 




def kl_divergence_loss(mean, variance, strategy_obj):
    with strategy_obj.scope():
        return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance), reduction=tf.keras.losses.Reduction.SUM)

def generator_loss(y, strategy_obj):
    with strategy_obj.scope():
        y = (y + 1.0) / 2.0
        return tf.reduce_mean(y, reduction=tf.keras.losses.Reduction.SUM)


class FeatureMatchingLoss(kr.losses.Loss):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mae = kr.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred, strategy_obj):
            with strategy_obj.scope():
                loss = 0
                y_true = [(y + 1.0) / 2.0 for y in y_true]
                y_pred = [(y + 1.0) / 2.0 for y in y_pred]
                #y_true = (y_true + 1.0) / 2.0
                #y_pred = (y_pred + 1.0) / 2.0
                
                for i in range(len(y_true) - 1):
                        loss += self.mae(y_true[i], y_pred[i])
                return loss


class VGGFeatureMatchingLoss(kr.losses.Loss):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.encoder_layers = [
                    "block1_conv1",
                    "block2_conv1",
                    "block3_conv1",
                    "block4_conv1",
                    "block5_conv1",
            ]
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            vgg = kr.applications.VGG19(include_top=False, weights="imagenet")
            layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
            self.vgg_model = kr.Model(vgg.input, layer_outputs, name="VGG")
            self.mae = kr.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred, strategy_obj):
            with strategy_obj.scope():
                y_true = (y_true + 1.0) / 2.0
                y_pred = (y_pred + 1.0) / 2.0
                
                y_true = tf.image.grayscale_to_rgb(y_true)
                y_pred = tf.image.grayscale_to_rgb(y_pred)
                
                y_true = kr.applications.vgg19.preprocess_input(127.5 * (y_true + 1))
                y_pred = kr.applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
                real_features = self.vgg_model(y_true)
                fake_features = self.vgg_model(y_pred)
                loss = 0
                for i in range(len(real_features)):
                        loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
                return loss


class DiscriminatorLoss(kr.losses.Loss):
    def __init__(self, strategy_obj, **kwargs):

        with strategy_obj.scope():
            super().__init__(**kwargs)
            self.hinge_loss = kr.losses.Hinge(reduction=tf.keras.losses.Reduction.NONE)


    def call(self, is_real, y_pred, strategy_obj):
        with strategy_obj.scope():
            label = 1.0 if is_real else -1.0
            return self.hinge_loss(label, y_pred)

class MAE(kr.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = kr.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred, strategy_obj):
        with strategy_obj.scope():
            y_true = (y_true + 1.0) / 2.0
            y_pred = (y_pred + 1.0) / 2.0
            return self.mae(y_true, y_pred)
    

'''Pix2Pix Losses'''
class p2p_disc_loss(kr.losses.Loss):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    def call(self, real_img, fake_img, strategy_obj):
            with strategy_obj.scope():
                real_loss = tf.reduce_mean(real_img, reduction=tf.keras.losses.Reduction.SUM)
                fake_loss = tf.reduce_mean(fake_img, reduction=tf.keras.losses.Reduction.SUM)
                return fake_loss - real_loss
