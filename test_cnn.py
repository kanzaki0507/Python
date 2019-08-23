import numpy as np
import glob
import keras
from keras.preprocessing.image import img_to_array, load_img
import pdb
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3


file_list = glob.glob('figs/*/*.jpg')
print(file_list)

X = []
Y = []
# 画像を読み込む
for file in file_list:
    print(file)

    img = img_to_array(load_img(file, grayscale=False, color_mode='rgb', target_size=(150, 150)))
    X.append(img)

    # label処理
    split_file = file.split('/')
    print(split_file)

    if split_file[1]=='dogs':
        Y.append(0)
    else:
        Y.append(1)

    # pdb.set_trace()

X = np.asarray(X)
print(X.shape, X.max(), X.min(), X.dtype)

Y = np.asarray(Y)
print(Y.shape, Y.dtype, Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print(y_train[1], Y_train[1])

# modelを組む
input_tensor = Input(shape=(150, 150, 3))  # this assumes K.image_data_format() == 'channels_last']
# VGG16だと入力 >= 48
# base_model = VGG16(include_top=False, input_tensor=input_tensor)

# inceptionV3だと入力 >= 
base_model = InceptionV3(include_top=False, input_tensor=input_tensor)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(200, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary()