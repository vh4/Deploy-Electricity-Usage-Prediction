#main program to running all function and classes
#Default => LSTM Deep Learning
#Author => Fathoni Waseso J

from threading import Thread
from uelib.predict import Predict
from uelib.uelectrict.api import Api
import time
import os

#memanggil class utama untuk menjalankan semua perintah dari child nya
#table lstm yang digunakan untuk upload data
#user id = user kamu

serialnumber    = "1234567892"
n_input = 4
#data actual
api = Api()
jumlah_data_aktual = len(api.GET(serialnumber))

print("data actual anda : " + str(jumlah_data_aktual))
print("")

def PelatihanModel():
    print("sedang running pada fungsi pelatihan model")
    while True:
        if jumlah_data_aktual < 500: #butuh 500 data untuk menjalankan fungsi pelatihan model
            print("masih dalam proses pengumpulan data untuk training. tunggu sampai data selesai !")
        else:
            model = Predict(serialnumber)
            model.fit()
            break

#eksekusi fungsi pelatihan model
PelatihanModel()