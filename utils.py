import pickle
import numpy as np
import torchvision
import torch

'''
CONSTANTS
'''
NUM_CHANNELS = 3
NUM_PIXELS_X = 32
NUM_PIXELS_Y = 32

'''
UTILITY FUNCTIONS
'''
def read_data(f_name):
    '''
    Extracts readable data from binary data file
    f_name - name of binary data file
    data - (num_points, num_channels, num_pixels_x, num_pixels_y) numpy array
    labels - (num_labels) list
    '''
    dict = unpickle(f_name)
    labels = np.array(dict[b'labels'])
    num_points = labels.shape[0]
    data = dict[b'data']
    data = data.reshape(num_points, NUM_CHANNELS, NUM_PIXELS_X, NUM_PIXELS_Y)
    return data, labels

def augment(data):
    '''
    data - one or more images as a (num_channels, num_pixels_x, num_pixels_y) numpy array
    Returns one or more augmented images of the same shape
    '''
    aa = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
    input_tensor = torch.tensor(data, dtype=torch.uint8)
    return aa.forward(input_tensor).numpy()

def gen_copies(data, num_copies=2):
    '''
    data - (num_points, num_channels, num_pixels_x, num_pixels_y) numpy array
    num_copies - number of versions of each image to create
    output - (num_points * num_copies, num_channels, num_pixels_x, num_pixels_y) numpy array
    '''
    num_points, _, _, _ = data.shape
    output = np.zeros((num_points * num_copies, NUM_CHANNELS, NUM_PIXELS_X, NUM_PIXELS_Y), dtype=np.uint8)
    for i in range(num_points):
        output[i*num_copies] = data[i]
        for j in range(num_copies-1):
            output[i*num_copies+j+1] = augment(data[i])
    return output

'''
HELPER FUNCTIONS
'''
def unpickle(f_name):
    with open(f_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
