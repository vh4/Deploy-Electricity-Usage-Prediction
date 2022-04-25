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
    time.sleep(10)
    print("sedang running pada fungsi pelatihan model")
    while True:
        if jumlah_data_aktual < 500:
            print("masih dalam proses pengumpulan data untuk training. tunggu sampai data selesai !")
            time.sleep(10)
        else:
            model = Predict(serialnumber)
            model.fit()
            time.sleep(1728000) #setiap 20 hari dia belajar
            
def PrediksiModel():
    time.sleep(3)
    print("sedang running pada fungsi prediksi model")
    while True:
        if jumlah_data_aktual < 24 + n_input:
            print("masih dalam proses pengumpulan data actual untuk prediksi!")
        else:
            model = Predict(serialnumber)
            model.predict()
            time.sleep(86400) #setiap  1 hari akan di prediksi

t2 = Thread(target = PrediksiModel)
t1 = Thread(target = PelatihanModel)

#t1.setDaemon(True)
#t2.setDaemon(True)

t1.start()
t2.start()
