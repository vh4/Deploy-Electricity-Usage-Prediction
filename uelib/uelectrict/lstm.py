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

        self.n_input = 5 # input for time series . input for neural network
        self.n_output_label = 1 # number of outputs from NN

        #units => output dimensions of neurons
        #Dense => number of hidden layers hidden layers

        #checking model is exist. if there is, the model that has been trained will be trained again to make it smarter!
        self.model = ''

        if(os.path.exists('./uelib/model/lstm.h5')):

            # load model with tensorflow and examples => see below the model that has been trained and is being trained again
            # the model will learn more and be smarter (the more it reduces the error value to as small as possible!)

            print(Fore.CYAN, "Model already exists, model training will be conducted to improve model performance!")
            print("")
            print(Style.RESET_ALL)

            self.model = load_model('./uelib/model/lstm.h5')

            #see the MAPE metric error -> weight and bias are not random and start from the results of the training above earlier.
            self.hasil = self.model.fit(X_train_for_predict, y_train_for_predict, epochs=150, batch_size=1)
        else:

            print(Fore.RED, ".h5 model does not exist, the proccess will do to make model !")
            print("")
            print(Style.RESET_ALL)
            time.sleep(1)
            self.model = Sequential([
                tf.keras.layers.LSTM(128, activation='relu', input_shape=(self.n_input, self.n_output_label)),
                Dense(1)
            ])

            #loss MSE untuk error function, optimizer yaitu sgd for optimasi weight dan bias
            #call model and training data

            self.model.compile(loss=['mse'], optimizer= tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=["mse"])
            print("")
            print(self.model.summary())

            #1 epochs => 1x all training data from top to bottom to completion.
            #fit generator to execute the model.

            self.hasil = self.model.fit(X_train_for_predict, y_train_for_predict,  epochs=150, batch_size=1)

        print("")

        return self.model


