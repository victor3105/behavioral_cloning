import cv2
import numpy as np
import csv
from math import ceil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def normalize_data(data):
    '''
    Normalize data and shift mean to zero
    '''
    data /= 255.0 - 0.5
    return data


def read_data(file_path, list_to_store=[]):
    '''
    Read lines from CSV file
    '''
    with open(file_path) as file:
        reader = csv.reader(file)
        for line in reader:
            list_to_store.append(line)
    return list_to_store
   

def prepare_samples(samples, batch_size=32):
    '''
    This generator prepares images and steering angle values
    for validation and training sets
    '''
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch = samples[offset : (offset + batch_size)]
            images = []
            measurements = []
            for line in batch:
                # read center images in
                img = cv2.imread(line[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                # read left and right images in
                left_img = cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2RGB)
                # read in steering angle value
                m = float(line[3])
                measurements.append(m)
                # flip image to augment data
                flipped_img = np.fliplr(img)
                flipped_measurement = -m
                images.append(flipped_img)
                measurements.append(flipped_measurement)
                # correction angle to learn the car to return to center
                correction = 0.2
                images.extend((left_img, right_img))
                measurements.extend(((m + correction), (m - correction)))
            
            x_train = np.array(images)
            y_train = np.array(measurements)
            
            yield shuffle(x_train, y_train)
    
files = ['../Data/Track1/driving_log.csv',
         '../Data/Track1_reversed/driving_log.csv']

data = []
for file in files:
    data = read_data(file, data)
    
train_data, validation_data = train_test_split(data, test_size=0.2)

batch_size = 128

train_gen = prepare_samples(train_data, batch_size)
valid_gen = prepare_samples(validation_data, batch_size)

# we will use NVIDIA's CNN architecture
d = 0.5
model = Sequential()
model.add(Lambda(normalize_data, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Dropout(d))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Dropout(d))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Dropout(d))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(d))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(d))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(d))
model.add(Dense(50))
model.add(Dropout(d))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, steps_per_epoch=ceil(len(train_data) * 4 / batch_size),
                    validation_data=valid_gen,
                    validation_steps=ceil(len(validation_data) * 4 / batch_size),
                    epochs=5, verbose=1)

model.save('model.h5')
print("Model's been saved")
