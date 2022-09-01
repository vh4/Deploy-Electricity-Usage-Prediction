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

serialnumber = "20220812-1"
n_input = 5
#data actual
api = Api()
jumlah_data_aktual = len(api.GET(serialnumber))

print("data actual anda : " + str(jumlah_data_aktual))
print("")

def PelatihanModel():
    print("is running on the model training function")
    while True:
        if jumlah_data_aktual < 29: #requires min 29 data
            os.system('cls')
            print("still in the process of collecting data for training. wait until the data is complete!")
            #time.sleep(60) # 1 minute
            break
        else:
            model = Predict(serialnumber)
            model.fit()
            print("done proccess training!")  
            #time.sleep(1728000) #each 20 days, models is training
            break
#execution of model prediction function
PelatihanModel()