import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import vgg16
import load
from torchsummary import summary

classes = ['plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def show_train_image():
    data = load_file('./cifar-10-batches-py/data_batch_1')
    image_data = data['data']
    labels = data['labels']
    label_count = len(labels)
    picture_data = image_data.reshape(-1,3,32,32)
    picture_data = picture_data.transpose(0,2,3,1)
    show_images_nums = 10


    for i in range(2, 6):
        batch_path = './cifar-10-batches-py/data_batch_' + str(i)
        data = load_file(batch_path)
        image_data = data['data']
        new_labels = data['labels']
        new_picture_data = image_data.reshape(-1,3,32,32)
        new_picture_data = new_picture_data.transpose(0,2,3,1)
        picture_data = np.append(picture_data, new_picture_data, axis=0)
        labels = np.append(labels, new_labels, axis=0)

    random_list = []
    train_len = len(picture_data)
    for i in range(show_images_nums):
        n = random.randint(1,train_len)
        if not n in random_list:
            random_list.append(n)

    row = 1
    fig,a =  plt.subplots(2,int(show_images_nums/2))
    for idx, rand_num in enumerate(random_list):
        if(idx == show_images_nums/2):
            row += 1
        col = (idx)%5
        a[row-1][col].imshow(picture_data[rand_num])
        a[row-1][col].set_title(classes[labels[rand_num]])
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.show()

# def show_params():
def show_model_structure():
    model = vgg16.load_model()
    summary(model, (3, 32, 32))

def show_model_structure():
    model = vgg16.load_model()
    summary(model, (3, 32, 32))