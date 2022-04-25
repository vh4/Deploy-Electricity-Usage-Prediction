import time
import requests
import tensorflow as tf
from colorama import init
init()
from tqdm import trange
from colorama import Style, Fore, Back

class Api:

    #mengambil API dari website
    #method http get merupakan sebuarh http url untuk mengambil / fetch data lewat API
    #json() => memanggil sebuah fungsi json untuk mengambil data yang bentuknya json dari API

    def GET(self, serialnumber):
        url = "https://u-elektrik.my.id/api/data/" + serialnumber
        response = requests.get(url)
        self.data_actual = response.json()

        return self.data_actual

    def POST(self, serialnumber, data):

        url = "https://u-elektrik.my.id/api/data/" + serialnumber

        requests.post(url, data=data)

        print("data berhasil di kirim !")