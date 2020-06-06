from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#1. convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation ='relu' ))

#2. Max pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#3. Flattening
classifier.add(Flatten())

#4.  fully connected layer
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#5. compilung CNN
classifier.compile(optimizer ='adam', loss='binary_crossentropy',metrics=['accuracy'])

#6. fitting CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                    'C:/Users/Albus Dumbledore/OneDrive/Desktop/DEsktop/machine learning/P14-Part8-Deep-Learning/Section 36 - Convolutional Neural Networks (CNN)/Python/dog vs cat/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'C:/Users/Albus Dumbledore/OneDrive/Desktop/DEsktop/machine learning/P14-Part8-Deep-Learning/Section 36 - Convolutional Neural Networks (CNN)/Python/dog vs cat/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit_generator(
                training_set,
                samples_per_epoch=8000,
                nb_epoch=25,
                validation_data=test_set,
                nb_val_samples=2000)

#result = loss: 0.2873 - accuracy: 0.8769 - val_loss: 0.5393 - val_accuracy: 0.7393
