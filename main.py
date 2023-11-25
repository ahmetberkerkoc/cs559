import tensorboard
import tensorflow as tf
# Load the TensorBoard notebook extension
import tensorflow as tf
import datetime
from keras import layers
import numpy as np
import cv2
import os
import numpy as np
import keras
from keras import Sequential, Model, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D



def custom_activation(x):
    return K.clip(x,0.0,10.0)

training_path = './SCUT_FBP5500_downsampled/training'
validation_path =  './SCUT_FBP5500_downsampled/validation'
test_path = './SCUT_FBP5500_downsampled/test'
x=[]
y=[]
x_val=[]
y_val=[]
for images in os.listdir(training_path):
    x.append(cv2.imread(training_path+'/'+images)/255.)
    y.append(float(images.split('_')[0]))

for images in os.listdir(validation_path):
    x_val.append(cv2.imread(validation_path+'/'+images)/255.)
    y_val.append(float(images.split('_')[0]))

X=np.array(x)
Y=np.array(y)
Xval=np.array(x_val)
Yval=np.array(y_val)


train_datagen = ImageDataGenerator()
train_datagen=train_datagen.flow(X,Y,batch_size=8)
val_datagen = ImageDataGenerator()
val_datagen=val_datagen.flow(Xval,Yval,batch_size=8)

print("here")




x_test=[]
y_test=[]

for images in os.listdir(test_path):
    x_test.append(cv2.imread(test_path+'/'+images)/255.)
    y_test.append(float(images.split('_')[0]))

Xtest=np.array(x_test)
Ytest=np.array(y_test)

test_datagen = ImageDataGenerator()
test_datagen=test_datagen.flow(Xtest,Ytest,batch_size=8)




model = Sequential([
    Input(shape=(80, 80, 3)),
    Conv2D(filters=6, kernel_size=(5, 5), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=120, kernel_size=(5, 5), padding="same", activation="relu"),
    Flatten(),
    Dense(units=84, activation="relu"),
    Dense(units=1, activation=custom_activation)
])

print(model.summary())


model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.001))


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history1 = model.fit(
    train_datagen,
    epochs=1,
    steps_per_epoch=Y.shape[0]// 8,
    validation_data=val_datagen,
    verbose=1,
    callbacks=[tensorboard_callback])



model.evaluate(
    test_datagen,
    verbose="auto"
)

print(model.predict(np.random.rand(1,80,80,3))[0][0])