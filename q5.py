import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import vgg16
from torchsummary import summary
import torch
import torchvision
import matplotlib.image as mpimg

classes = ['plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_params():
    batch_size, lr = vgg16.get_params()
    print("Batch size: %d" % batch_size)
    print("Learning rate: %f" % lr)
    print("Optimizer: SGD")

def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def show_train_image():
    data = load_file('./data/cifar-10-batches-py/data_batch_1')
    image_data = data['data']
    labels = data['labels']
    label_count = len(labels)
    picture_data = image_data.reshape(-1,3,32,32)
    picture_data = picture_data.transpose(0,2,3,1)
    show_images_nums = 10


    for i in range(2, 6):
        batch_path = './data/cifar-10-batches-py/data_batch_' + str(i)
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

def test_image(index):
    try:
        index = int(index)
    except ValueError:
        # Handle the exception
        print('Please enter an integer')
        return 
    if index < 0 or index > 10000:
        print("wrong index range")
        return

    print("\n... Testing ...")
    print("Test Image: %d" % index)
    model = vgg16.load_model()
    testloader = vgg16.get_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    total = 0
    correct = 0
    with torch.no_grad():
        test_iter = iter(testloader)
        global images, labels
        for i in range(index+1):
            images, labels = test_iter.next()

        images, labels = images.to(device), labels.to(device)
        output = model(images)
        m = nn.Softmax(dim=1)
        output = m(output)
        output = output.cpu().numpy()[0]

        # Make a fake dataset:
        y_pos = np.arange(len(classes))
        
        # Create bars
        plt.bar(y_pos, output)
        
        # Create names on the x-axis
        plt.xticks(y_pos, classes)
        
        # Show graphic
        # plt.imshow(images.cpu()[0].permute(1, 2, 0))
        plt.show()

        fig1, ax1 = plt.subplots(figsize=(3,3), dpi=50)
        ax1.imshow(images.cpu()[0].permute(1, 2, 0))
        plt.show()

def show_accuracy():
    acc = mpimg.imread('result/accuracy.png')
    loss = mpimg.imread('result/loss.png')
    fig, ax = plt.subplots()
    imgplot = plt.imshow(acc)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()
    fig1, ax1 = plt.subplots()
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.imshow(loss)
    plt.show()