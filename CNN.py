!pip install tensorflow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


print("x_train_shape", x_train.shape)
print("y_train_shape", y_train.shape)
print("x_test_shape", x_test.shape)
print("y_test_shape", y_test.shape)

plt.rcParams['figure.figsize'] = (9,9)

for i in range(9):
    plt.subplot(3,3,i+1)
    num = random.randint(0, len(x_train))
    plt.imshow(x_train[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[num]))

plt.tight_layout()

x_train[0].shape
x_trains = x_train.reshape(60000, 784) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
x_train.shape

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

matprint(x_train[num])

x_trains = x_train.reshape(60000, 784) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
x_tests = x_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.

x_trains = x_trains.astype('float32')   # change integers to 32-bit floating point numbers
x_tests = x_tests.astype('float32')

x_trains /= 255                        # normalize each value for each pixel for the entire vector for each input
x_tests /= 255

print("Training matrix shape", x_trains.shape)
print("Testing matrix shape", x_tests.shape)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3),
activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3),
activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3),
activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True),

metrics=['accuracy'])

history = model.fit(x_trains.reshape(-1, 28, 28, 1),
y_train, epochs=10,
validation_data=(x_tests.reshape(-1, 28, 28, 1),
y_test))

test_loss, test_acc = model.evaluate(
    x_tests.reshape(-1, 28, 28, 1), y_test, verbose=2)
print('Test accuracy:', test_acc)

from google.colab import drive
drive.mount('/content/drive')
model.save('/content/drive/My Drive/model.hdf5')
# model.save('/content/drive/My Drive/mon_modele')

# Take an image from the test set (for example, the first image)
num = int(input('Give the index of the image in the range from 0 to 9999 to predict \n'))
img = x_test[num]
label = y_test[num]  # Obtenez l'étiquette réelle pour la comparaison


# Make predictions
predictions = model.predict(x_tests[num].reshape(-1, 28, 28, 1))
classe_predite = np.argmax(predictions, axis=1)

# Display the image and the prediction
plt.imshow(img, cmap='gray')  # Using the original image
plt.title(f"Prédiction : {classe_predite[0]}, Étiquette réelle : {label}")
plt.axis('off')
plt.show()

import sys
print(sys.version)

