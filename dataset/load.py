import os,sys
import requests
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.datasets import *

class Load():
    def __init__(self, name):
        if name == "kuzushiji":
            self.get_kuzushiji()
        else:
            self.name = 'tf.keras.datasets.'+ name
            self.datasets = eval(self.name)
            if name == 'mnist':
                self.size, self.channel = 28, 1
                self.output_dim = 10
            elif name == 'cifar10':
                self.size, self.channel = 32, 3
                self.output_dim = 10
            elif name == 'cifar100':
                self.size, self.channel = 32, 3
                self.output_dim = 100
            else:
                NotImplementedError
    
    def load(self):
        try:
            return self.datasets.load_data(label_mode='fine')
        except:
            return self.datasets.load_data()

    def get_kuzushiji(self):
        if not os.path.isfile('./dataset/k49-train-imgs.npz'):
            self.down_load_kuzushiji()
        train_image = np.load('./dataset/k49-train-imgs.npz')
        train_label = np.load('./dataset/k49-train-labels.npz')
        test_image = np.load('./dataset/k49-test-imgs.npz')
        test_label = np.load('./dataset/k49-test-labels.npz')
        self.x_train = train_image['arr_0']
        self.y_train = train_label['arr_0']
        self.x_test = test_image['arr_0']
        self.y_test = test_label['arr_0']
        self.size, self.channel = 28, 1
        self.output_dim = 49

    def down_load_kuzushiji(self):
        url_list = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
                    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
                    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
                    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz']
        for url in url_list:
            path = url.split('/')[-1]
            r = requests.get(url, stream=True)
            with open("./dataset/"+path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk)