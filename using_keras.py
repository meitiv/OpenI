#!/usr/bin/env python3

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Flatten())  # this converts our 3D feature maps to 1D
# feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# save model
from keras.models import model_from_json
with open('keras_model.json', 'w') as f:
    f.write(model.to_json())

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('OpenI',
                                                    target_size=(150, 150),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory('OpenI',
                                                  target_size=(150, 150),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=2000 // batch_size,
                    epochs=50,
                    validation_data=train_generator,
                    validation_steps=800 // batch_size)
model.save_weights('first_try_keras.h5')
