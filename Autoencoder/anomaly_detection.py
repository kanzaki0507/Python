from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
 
# AutoEncoder ネットワーク構築
encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
 
# MNIST データ読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
  # 学習データを「１」のみにする
x1 =[]
for i in range(len(x_train)):
    if y_train[i] == 1 :
       x1.append(x_train[i])
x_train = np.array(x1)
 
# テストデータを「１」と「９」にする
x2, y = [],[]
for i in range(len(x_test)):
    if y_test[i] == 1 or y_test[i] == 9 :
       x2.append(x_test[i])
       y.append(y_test[i])
x_test = np.array(x2)
y = np.array(y)


# データの前準備
x_train, x_valid = train_test_split(x_train, test_size=0.175)
x_train = x_train.astype('float32')/255.
x_valid = x_valid.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
 
# 学習
autoencoder.fit(x_train, x_train,
                nb_epoch=300,
                batch_size=256,
                shuffle=True,
                validation_data=(x_valid, x_valid))
 
# 出力画像の取得
decoded_imgs = autoencoder.predict(x_test)
 
# サンプル画像表示
n = 6
plt.figure(figsize=(12, 6))
for i in range(n):
    # テスト画像を表示
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
    # 出力画像を表示
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # 入出力の差分画像を計算
    diff_img = x_test[i] - decoded_imgs[i]
    
    # 入出力の差分数値を計算
    diff = np.sum(np.abs(x_test[i]-decoded_imgs[i]))
        
    # 差分画像と差分数値の表示
    ax = plt.subplot(3, n, i+1+n*2)
    plt.imshow(diff_img.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True) 
    ax.set_xlabel('score = '+str(diff))    
        
plt.savefig("result.png")
plt.show()
plt.close()