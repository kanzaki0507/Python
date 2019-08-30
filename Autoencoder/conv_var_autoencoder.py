from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras,datasets import mnist
from keras.losses import binary_crossentropy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

latent_dim = 32
batch_size = 64
img_size = 28
epochs = 5

# encoder
input_img = Input(shape=(img_size, img_size, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 潜在変数
z_mean = Dense(latent_dim, name='z_mean')(encoded)
z_log_var = Dense(latent_dim, name='z_log_var')(encoded)

def sampling(args):
    z_mean, z_log_var = args
    latent_dim = 32
    epsilon_std = 1.0
    epsilon = K.random_nomal(shape=(K.shape(z_mean)[0],
                                    K.shape(z_mean)[1], 
                                    K.shape(z_mean)[2], 
                                    latent_dim),
                            mean = 0.,
                            stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(z)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoderの定義
autoencoder = Model(inpup_img, decoded)

# loss
# Compute VAE loss