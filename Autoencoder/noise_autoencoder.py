import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np

batch_size = 32
num_classes = 10
epochs = 100
saveDir = "/home/kanzaki/python/Autoencoder/"
if not os.path.isdir(saveDir):
  os.makedirs(saveDir)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_val = x_test[:7000]
x_test = x_test[7000:]
print("validation data: {0} \ntest data: {1}".format(x_val.shape, x_test.shape))

# add noise to data
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0., scale=1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0., scale=1.0, size=x_test.shape)
x_val_noisy = x_val + noise_factor * np.random.normal(loc=0., scale=1.0, size=x_val.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
x_val_noisy = np.clip(x_val_noisy, 0., 1.)

# show noisy images
# definition to show original image and reconstructed image
def showOrigDec(orig, noise, num=10):
  import matplotlib.pyplot as plt
  n=num
  plt.figure(figsize=(20, 4))
  
  for i in range(n):
    # desplay original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(orig[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display original
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(noise[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

showOrigDec(x_train, x_train_noisy)