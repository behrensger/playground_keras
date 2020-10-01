#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:59:06 2019

@author: abehrens
"""
#import numpy as num
#import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#import PIL.Image as pil

#Daten
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


network = models.Sequential()

#Network-Layers
network = models.Sequential()
network.add(layers.Dense(400, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(400))
#network.add(layers.Dense(200))
#network.add(layers.Dense(100))
#network.add(layers.Dense(50))
#network.add(layers.Dense(25))
network.add(layers.Dense(10, activation='softmax'))


#Festlegung Optimierer und Verlustfunktion
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#Trainingsdaten vorverarbeiten (wertebereich 0..1)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Netzwerk anlernen
fit_history = network.fit(train_images, train_labels, epochs=6, batch_size=128, verbose=0)


#Test the network
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=0 )
print('test_acc:', test_acc, 'test_loss', test_loss)

#Ausgabe des Fit-Verlaufs
#history_dict = fit_history.history
history_dict = fit_history.history
print(history_dict.keys())
loss_values = history_dict['loss']
accu_values = history_dict['accuracy']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Verlust Training')
plt.plot(epochs, accu_values, 'b', label='Verlust Validierung')
plt.title('Wert der Verlustfunktion Training/Validierung')
plt.xlabel('Epochen')
plt.ylabel('Wert der Verlustfunktion')
plt.legend()
plt.show() 
