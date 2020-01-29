import csv
import cv2
import matplotlib.pyplot as mp
import keras
import numpy as np
import sklearn
import os
from random import shuffle
from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Lambda,Dropout
from keras.layers import Cropping2D , BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import ELU
from keras.models import load_model

from sklearn.model_selection import train_test_split

samples = []
samples1 = []



with open('/home/carnd/aws_share/data/driving_log.csv') as csvfile:
    
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples1.append(line)
        
        
        
for i in range(0,15): # iteration to increase the occurance of the images of the bridge in the first track     
    for i in samples1[72:182]:
        samples.append(i)
    
        
# Below iterations are done to increase the occurance of the sharp edges       

        
for i in range(0,4):     
    for i in samples1:
        if (float(i[3]) < 0.0):
            samples.append(i)
            
for i in range(0,25):     
    for i in samples1:
        if (float(i[3]) <= -0.3):
            samples.append(i)
            
            
for i in range(0,25):   
    for i in samples1:
        if (float(i[3]) >= 0.3):
            samples.append(i)
            
            
            
for i in range(0,60):    
    for i in samples1:
        if (float(i[3]) >= 0.6):
            samples.append(i)
            
            
            
for i in range(0,25):       
    for i in samples1:
        if (float(i[3]) <= -0.6):
            samples.append(i)
       
       


for i in samples1:
    samples.append(i)
               
               

        
# histogram function is used to whether the data contains more or less equal occurance of all range of steering angles     
steer  = []

for i in samples:
    
    steer.append(float(i[3]))
    

print(np.min(steer))
print(np.max(steer))
array1,array = np.histogram(steer,bins= [-1,-0.5,-0.25,0,0.25,0.5,1]) 
    
print(array)   
print(array1)

# Splitting the data set into train,validation and test images.
    
train_samples, tests = train_test_split(samples, test_size=0.05) 
validation_samples, test_samples = train_test_split(tests, test_size=0.2)

print(len(train_samples)) 
print(len(validation_samples))
print(len(test_samples))


# Python Generator Function


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        a = np.array(samples)
        shuffle(a)
        b = np.array(a)
        shuffle(b)
        c = np.array(b)
        shuffle(c)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_samples1 = a[offset:offset+batch_size]
            batch_samples2 = b[offset:offset+batch_size]
            batch_samples3 = c[offset:offset+batch_size]
            images = []
            angles = []
            for i in batch_samples:
                
                path = i[0]
                path = path.split('/')[-1]
                path  = '/home/carnd/aws_share/data/IMG/' + path
             
                image = cv2.imread(path)
      
                images.append(image)
         
                center_angle1 = float(i[3])
      
                angles.append(center_angle1)
           
                
            for i in batch_samples1:
                
                path = i[0]
                path = path.split('/')[-1]
                path  = '/home/carnd/aws_share/data/IMG/' + path
         
                image = cv2.imread(path)
                image2 = np.fliplr(image) # To flip the image
                images.append(image2)
                center_angle2 = - float(i[3])
                angles.append(center_angle2)
                    
               
                        
            for i in batch_samples2:
               
                path = i[1]
                path1 = len(path)
                
                
                
                path = path.split('/')[-1]
                path  = '/home/carnd/aws_share/data/IMG/' + path
                image = cv2.imread(path)  # using left camera image 
                images.append(image)
                measurement = float(i[3]) + 0.2  # steering with correction for left camera
                angles.append(measurement)
                    
                
                    
                
                
            for i in batch_samples3:
                path = i[2]
                path1 = len(path)
                
              
                
                path = path.split('/')[-1]
                path  = '/home/carnd/aws_share/data/IMG/' + path
                image = cv2.imread(path) # using right camera image 
                images.append(image)
                measurement = float(i[3]) - 0.2 # steering with correction for right camera
                angles.append(measurement)
                
                
                
                    
            
           
            
            

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size= 32)
validation_generator = generator(validation_samples, batch_size= 32)

#defining the model in keras

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3))) # Cropping the top of the image
model.add(Lambda(lambda x : (x/255.0 - 0.5))) # Normalizing the dataset

model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2),padding='valid')) # Convolutional layer1
model.add(ELU()) # ELU function to achieve Nonlinearity
model.add(Dropout(0.5)) # Dropout to avoid overfitting

model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2),padding='valid')) # Convolutional layer2
model.add(ELU())
model.add(Dropout(0.5))

model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2),padding='valid')) # Convolutional layer3 
model.add(ELU())
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=1,epsilon=0.001)) 


model.add(Conv2D(64,kernel_size=(3,3),padding='valid'))# Convolutional layer4
model.add(ELU())
model.add(Dropout(0.5))


model.add(Conv2D(64,kernel_size=(3,3),padding='valid'))# Convolutional layer5
model.add(ELU())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(BatchNormalization(epsilon=0.001))

model.add(Dense(100)) # Fully connected layer1
model.add(ELU())
model.add(Dropout(0.5))

model.add(Dense(50)) # Fully connected layer2
model.add(ELU())
model.add(Dropout(0.5))

model.add(Dense(10)) # Fully connected layer3
model.add(ELU())

model.add(Dense(1)) # Final output
model.add(ELU())

model.compile(loss='mse',optimizer='adam')

model = load_model('/home/carnd/aws_share/model3.h5')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//32, epochs = 5 , verbose=1, validation_data=validation_generator , validation_steps = 2, shuffle = True)

# Saving the trained model

model.save('/home/carnd/aws_share/model5.h5')


           





