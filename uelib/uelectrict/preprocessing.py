#file name => Preprocessing
#Author => fathoni waseso jati
#description => preprocessing data - sebelum masuk kedalam model

import pandas as pd
import time
from colorama import init
from colorama import Style, Fore, Back
import numpy as np
import os

init()

class Preprocessing():

    def transformation(self, data):

        #mentransfromasikan data json kedalam dataframe yang berisi kolom dan baris
        #mengconvert tipe data object ke float
        #mengconvert tipe data object ke datetime

        self.df = pd.DataFrame(data)
        self.df["pemakaian_listrik"] = pd.to_numeric(self.df["pemakaian_listrik"],
                                                downcast="float")
        self.df['tanggal'] = pd.to_datetime(self.df["tanggal"])

        #membulatkan tanggal. misalkan tanggal 1-1-2020 12:12:12 -  12-1-2020 21:05:11 jadi 1-1-2020 12:00:00 - 12-1-2020 21:00:00
        self.df['tanggal'] = pd.Series(self.df['tanggal']).dt.round("H")

        #mengambil 2 kolom saja
        #membuat index yang awalnya integer menjadi date waktu.

        self.data_df = self.df[['tanggal', 'pemakaian_listrik']]
        self.data_df = self.data_df.set_index('tanggal')

        #mengisi tanggal dan data yang kosong ( reindex the data )
        self.my_range = pd.date_range(start=self.data_df[:1].index[0], end=self.data_df[-1:].index[0], freq='H')
        self.data_df = self.data_df.reindex(self.my_range, fill_value=None)

        return self.data_df

    def missingvalue(self,pemakaian_data):

        self.pemakaian_data = pemakaian_data

        # check apakah ada missing value atau tidak
        try:
            if self.pemakaian_data['pemakaian_listrik'].isna().sum() > 0:

                print("- Terdapat data yang kosong, mengisi data dengan rata-rata !")
                print(Style.RESET_ALL)

                time.sleep(1)

                # mengisi data yang kosong dengan nilai rata-rata keseluruhan data
                self.pemakaian_data['pemakaian_listrik'] = self.pemakaian_data['pemakaian_listrik'].fillna(
                    self.pemakaian_data['pemakaian_listrik'].mean())

                self.pemakaian_data['pemakaian_listrik'] = self.pemakaian_data['pemakaian_listrik'].mask(self.pemakaian_data['pemakaian_listrik']==0).fillna(self.pemakaian_data['pemakaian_listrik'].mean())

                print(Fore.GREEN, "- Data NULL berhasil ditambahkan !")
                print(Style.RESET_ALL)

            else:
                pass
        except:
            print(Back.RED, '- terjadi kesalahan pada data !!')
            print(Style.RESET_ALL)

        #menampilkan 5 data awal yang sudah diresampling
        print(Style.RESET_ALL)
        return self.pemakaian_data

    def slidingwindow(self, data_window, window_size):

        #convert dataframe kedalam numpy (matrik)
        data_to_numpy = data_window.to_numpy()

        Input  = []
        Output = []

        for i in range(len(data_to_numpy) - window_size): #misal data 5, windows_size=3, maka data sequence nya = 2

            #looping data_to numpy, misal data [1,2,3,4,5]. window 2 => 123, 234, 345
            row = [[a] for a in data_to_numpy[i:i + window_size]]

            #lalu ditampung di array variabel input
            Input.append(row)

            #sama seperti diatas. tapi tidaka da windows karena untuk output.
            label = data_to_numpy[i + window_size]

            #lalu ditampung di array variabel output
            Output.append(label)

        return np.array(Input), np.array(Output)

    def test_train_split_for_predict(self, data_splt_trn, window_size=4):

        #window size => input data yang akan digunakan untuk training neural network,
        data_to_sequence_for_predict = data_splt_trn['pemakaian_listrik']
        input_for_predict, output_for_predict = self.slidingwindow(data_to_sequence_for_predict, window_size=4)

        #X_actual => data 24 jam yang lalu untuk prediksi 24 jam kedepan.
        #y_actual => untuk output data actual. misal input 24 => output harus 24 juga karena data bersifat sequence to sequence

        #kenapa data actual hanya diprediksi 24 jam saja ? karena untuk menghindari ketika hari perkiraan hari besok ada teman yang datang dan penggunaan meningkat.
        #data xt1 sampai lag xt-24 = 24 data yang lam => akan diprediksi untuk xt1 sampai lag xt+24

        X_actual_for_predict, y_actual_for_predict = input_for_predict[-24:], output_for_predict[-24:]

        print(X_actual_for_predict.shape, y_actual_for_predict.shape)
        print("")

        return X_actual_for_predict, y_actual_for_predict

    def test_train_split_for_training(self, data_splt_trn, window_size=4):

        #window size => input data yang akan digunakan untuk training neural network,
        data_to_sequence_for_training = data_splt_trn['pemakaian_listrik']
        input_for_training, output_for_training = self.slidingwindow(data_to_sequence_for_training, window_size=4)

        #splitting data
        #X_train => untuk input data training dari seluruh data
        #dari program ini parameter model sudah ditentukan yang paling bagus, maka tidak perlu lagi data val untuk menguji overfit dan underfit. soalnya sudah di uji modelnya parameter.

        X_train_for_training, y_train_for_training = input_for_training[-418:], output_for_training[-418:]

        print(X_train_for_training.shape, y_train_for_training.shape)
        print("")

        return X_train_for_training, y_train_for_training
