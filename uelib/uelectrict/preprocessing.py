#file name => Preprocessing
#Author => fathoni waseso jati
#description => preprocessing data - before make model or predict model

import pandas as pd
import time
import numpy as np

class Preprocessing():

    def transformation(self, data):

        #transform json data into a dataframe containing columns and rows
        #convert object data type to float
        #convert object data type to datetime

        self.df = pd.DataFrame(data)
        self.df["pemakaian_listrik"] = pd.to_numeric(self.df["pemakaian_listrik"],
                                                downcast="float")
        self.df['tanggal'] = pd.to_datetime(self.df["tanggal"])

        #round (membulatkan) the date. example date 1-1-2020 12:12:12 -  12-1-2020 21:05:11 jadi 1-1-2020 12:00:00 - 12-1-2020 21:00:00
        self.df['tanggal'] = pd.Series(self.df['tanggal']).dt.round("H")

        #takes 2 columns only
        #make index yang awalnya integer to date waktu / convert to timeseries data

        self.data_df = self.df[['tanggal', 'pemakaian_listrik']]
        self.data_df = self.data_df.set_index('tanggal')

        #mengisi tanggal dan data yang kosong ( reindex the data )

        self.my_range = pd.date_range(start=self.data_df[:1].index[0], end=self.data_df[-1:].index[0], freq='H')
        self.data_df = self.data_df.reindex(self.my_range, fill_value=None)

        return self.data_df[-29:]

    def missingvalue(self,pemakaian_data):

        self.pemakaian_data = pemakaian_data

        #check whether there is a missing value or not
        try:
            if self.pemakaian_data['pemakaian_listrik'].isna().sum() > 0:

                print("- There is data that is empty, fill in the data with the average !")

                time.sleep(1)

                # fill in the blank data with the average value of the entire data
                self.pemakaian_data['pemakaian_listrik'] = self.pemakaian_data['pemakaian_listrik'].fillna(
                    self.pemakaian_data['pemakaian_listrik'].mean())

                self.pemakaian_data['pemakaian_listrik'] = self.pemakaian_data['pemakaian_listrik'].mask(self.pemakaian_data['pemakaian_listrik']==0).fillna(self.pemakaian_data['pemakaian_listrik'].mean())

                print("- NULL data added successfully !")

            else:
                pass
        except:
            print('- Data error occurred !!')

        return self.pemakaian_data

    def slidingwindow(self, data_window, window_size):

        #convert dataframe into numpy (matrik)
        data_to_numpy = data_window.to_numpy()

        Input  = []
        Output = []

        for i in range(len(data_to_numpy) - window_size): #example data 5, windows_size=3, so, data sequence nya = 2

            #looping data_to numpy, misal data [1,2,3,4,5]. window 2 => 123, 234, 345
            row = [[a] for a in data_to_numpy[i:i + window_size]]

            #then is accommodated in the input variable array
            Input.append(row)
            
            #Same as above. but no windows due to output (sama seperti diatas. tapi tidak ada windows karena untuk output)
            label = data_to_numpy[i + window_size]

            #then is accommodated in the output variable array
            Output.append(label)

        return np.array(Input), np.array(Output)

    def test_train_split_for_predict(self, data_splt_trn, window_size=5):

        #window size => #Same as above. but no windows due to output.
        data_to_sequence_for_predict = data_splt_trn['pemakaian_listrik']
        input_for_predict, output_for_predict = self.slidingwindow(data_to_sequence_for_predict, window_size=5)

        #X_actual => data from the past 24 hours for predictions for the next 24 hours.
        #y_actual => to output actual data. for example input 24 => output must be 24 as well because the data is sequence to sequence

        #why is the actual data only predicted 24 hours?
        #karena untuk menghindari ketika hari perkiraan hari besok ada teman yang datang dan penggunaan meningkat.
        #data xt1 sampai lag xt-24 = 24 data yang lag => akan diprediksi untuk xt1 sampai lag xt+24

        X_actual_for_predict, y_actual_for_predict = input_for_predict[-24:], output_for_predict[-24:]

        print(X_actual_for_predict.shape, y_actual_for_predict.shape)
        print("")

        return X_actual_for_predict, y_actual_for_predict

    def test_train_split_for_training(self, data_splt_trn, window_size=5):

        #window size => input data to be used for neural network training,
        data_to_sequence_for_training = data_splt_trn['pemakaian_listrik']
        input_for_training, output_for_training = self.slidingwindow(data_to_sequence_for_training, window_size=5)

        #splitting data
        #X_train => to input training data from all data
        
        #dari program ini parameter model sudah ditentukan yang paling bagus, 
        #maka tidak perlu lagi data val untuk menguji overfit dan underfit. soalnya sudah di uji (dibuat ) model parameter.

        X_train_for_training, y_train_for_training = input_for_training[-418:], output_for_training[-418:]

        print(X_train_for_training.shape, y_train_for_training.shape)
        print("")

        return X_train_for_training, y_train_for_training
