#main program to running all function and classes
#Default => LSTM Deep Learning
#Author => Fathoni Waseso J

from uelib.predict import Predict
from uelib.uelectrict.api import Api
import time
import os

#call the main class to execute all commands from its child
#table lstm used to upload data
#user id = your user id

serialnumber = "ASBZY-KMY6D-5QGYK-OBO9W"
n_input = 5
#data actual
api = Api()
jumlah_data_aktual = len(api.GET(serialnumber))

print("data actual : " + str(jumlah_data_aktual))
print("")

def PrediksiModel():
    print("is running on the model prediction function")
    while True:
        if jumlah_data_aktual < 24 + n_input: #need min 28 data to run the program
            os.system('cls')
            print("still in the process of collecting actual data for prediction!")
            #time.sleep(60)
            break
        else:
            model = Predict(serialnumber)
            model.predict()
            print("done proccess prediction!")
            break
            #time.sleep(86400) #each  1 days akan di prediksi
#execution of model prediction function
PrediksiModel()