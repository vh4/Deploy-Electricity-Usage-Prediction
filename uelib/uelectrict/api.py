import time
import requests
import tensorflow as tf
from colorama import init
init()
from tqdm import trange
from colorama import Style, Fore, Back

class Api:

    #fetch API from website
    #method http get is an http url to fetch / fetch data via API
    #json() => call a json function to fetch json data from API

    def GET(self, serialnumber):
        url = "https://u-elektrik.my.id/api/data/" + serialnumber
        response = requests.get(url)
        self.data_actual = response.json()

        return self.data_actual

    def POST(self, serialnumber, data):

        url = "https://u-elektrik.my.id/api/data/" + serialnumber

        requests.post(url, data=data)

        print("data successfully sended !")