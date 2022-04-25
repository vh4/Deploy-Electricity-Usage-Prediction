#Author      => Fathoni Waseso Jati
#Python      => 3.9.1
#Library     => Tensorflow
#uelectrict      => Long Short-Term Memory
#Description => Thesis Trial (S1) and PDT Penelitian dosen

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from colorama import init
from tqdm import trange
from colorama import Style, Fore, Back
import time
import os
init()

class LSTM():

    def train(self, X_train_for_predict, y_train_for_predict): #default parameter

        self.n_input = 4
        self.n_output_label = 1


        # input untuk timeseries
        # jumlah output dari NN

        #rumus MAPE yang sebenarnya dibawah ini !

        #units => dimensi keluaran dari neuron
        #Dense => jumlah hidden layer hidden layer

        #checking model is exis. if ada maka model yang sudah ditraining, akan di lakukan training lagi supaya lebih pintar !

        self.model = ''

        if(os.path.exists('./uelib/model/lstm.h5')):

            # load model with tensorflow dan contoh => lihat dibawah ini model yang sudah di training dan di lakukan training lagi
            # model akan semakin belajar dan semakin pintar(semakin mereduksi error value nya sampai sekecil mungkin !)

            print(Fore.CYAN, "model sudah ada, akan dilakukan training model untuk meningkatkan performa model !")
            print("")
            print(Style.RESET_ALL)

            self.model = load_model('./uelib/model/lstm.h5')

            # lihat MAPE metrik erronya -> weight dan biasnya sudah tidak random dan dimulai dari hasil training tadi diatas awal.

            self.hasil = self.model.fit(X_train_for_predict, y_train_for_predict, epochs=200,batch_size=4)
        else:

            print(Fore.RED, "model .h5 tidak ada, akan melakukan pembuatan model awal !")
            print("")
            print(Style.RESET_ALL)
            time.sleep(1)
            self.model = Sequential([
                tf.keras.layers.LSTM(64, activation='relu', input_shape=(self.n_input, self.n_output_label)),
                Dense(1)
            ])

            #loss MSE untuk error function, optimizer yaitu adam untuk optimasi bobot dan bias (backpropogation)
            #panggil model and training data

            self.model.compile(loss=['mse'], optimizer='adam', metrics=["mse", "mape"])
            print("")
            print(self.model.summary())

            #1 epochs => 1x seluruh data training dari atas ke bawah sampai selesai.
            #fit generator untuk mengeksekusi model.

            self.hasil = self.model.fit(X_train_for_predict, y_train_for_predict,  epochs=200, batch_size=4)

        print("")

        return self.model


