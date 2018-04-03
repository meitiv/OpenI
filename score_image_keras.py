#!/usr/bin/env python3

import sys
if len(sys.argv) != 3:
    print('Usage', sys.argv[0], 'model_json', 'weights_h5')
    sys.exit(1)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

# read the model
with open(sys.argv[1]) as f:
    model = model_from_json(f.read())

# read the weights
model.load_weights(sys.argv[2])

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# evaluate 
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory('OpenI',
#                                                   target_size=(150, 150),
#                                                   batch_size=16,
#                                                   class_mode='binary')
# score = model.evaluate_generator(test_generator)
# print (score)

import os
from PIL import Image
import numpy as np
import scipy.misc
# image size for model
size = 150,150
directory = [ 'OpenI/No/', 'OpenI/Yes/' ]
for idx, cla in enumerate(directory):
    for filename in os.listdir(cla):
        # read the image
        img = Image.open(cla + filename)
        # make an array, resize it, and rescale values to [0:1] range for
        # model evaluation
        img = np.array(img).astype(float)
        img *= 1./255.
        img = scipy.misc.imresize(img,size)
        # make a rank 4 tensor for model input
        img = np.expand_dims(img, axis = 0)
        # predict class
        pred = model.predict(img)
        print (idx, pred[0][0])
