from datetime import datetime
import tensorflow as tf
from pandas.tseries.offsets import DateOffset
from pathlib import Path
import numpy as np
import pandas as pd
from colorama import Style, Fore, Back
import time
from keras.models import load_model
import os
from uelib.uelectrict.lstm import LSTM
from uelib.uelectrict.api import Api
from uelib.uelectrict.preprocessing import Preprocessing
import json

class Predict(Api, Preprocessing, LSTM):
    def __init__(self, serialnumber):

           
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.serialnumber = serialnumber
        self.window_size = 5

        #1. fetch data Api method GET function
        self.data_dataframe = self.GET(self.serialnumber)

        #2. transformation data function
        self.data_dataframe = self.transformation(self.data_dataframe)

        #3. checking missing value funtion (clening data)
        self.data_dataframe = self.missingvalue(self.data_dataframe)

    def fit(self):

        #4. make sequence data function
        X_train_data_for_training, y_train_data_for_training  = self.test_train_split_for_training(self.data_dataframe, self.window_size)

        #5. training
        model_training = self.train(X_train_data_for_training, y_train_data_for_training)

        #6. Save
        print("sedang save model...")
        model_training.save('./uelib/model/lstm.h5')

    def predict(self):

        #spliting input shape for data prediction
        X_actual_data_for_predict, y_actual_data_for_predict = self.test_train_split_for_predict(self.data_dataframe, self.window_size)

        #laod model
        model_prediksi = load_model('./uelib/model/lstm.h5')

        #prediksi xt+24
        prediksi = model_prediksi.predict(X_actual_data_for_predict).flatten()
        os.system('cls')
        hasil_akhir = pd.DataFrame(data={'pemakaian_listrik': prediksi})

        #make range date tomorrow / kedepan
        tanggal_kedepan = [self.data_dataframe.index[-1] + DateOffset(hours=x) for x in
                           range(1, 25)]  # membuat dat baru dengan timestamp

        #concatenate (menggabungkan) the date dan result of prediction
        prediksi_df = pd.DataFrame(index=tanggal_kedepan)  # make dataframe columns dan rows.
        prediksi_df['pemakaian_listrik'] = hasil_akhir[
            'pemakaian_listrik'].values  #create a prediction_result column for predictive data that has been transformed

        prediksi_df = prediksi_df.reset_index()
        prediksi_dfs = prediksi_df.rename(columns={'index': 'tanggal'})
        prediksi_dfs['tanggal'] = prediksi_dfs['tanggal'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        prediksi_dfs['nomorserial_id'] = self.serialnumber

        #dataframe to json
        dataframe_to_json = prediksi_dfs.to_json(orient="records")

        #laod from dataframe json to json library format
        load_json = json.loads(dataframe_to_json)
        #dump

        #kirim ke database data hasil prediksi
        self.POST(self.serialnumber, json.dumps(load_json))