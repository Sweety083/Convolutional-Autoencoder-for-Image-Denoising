Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
An unsupervised artificial neural network called an autoencoder is trained to replicate its input into its output.

The image will be encoded into a lower-dimensional representation by an autoencoder, which will subsequently decode the representation back to the original image.

To obtain an identical output to the input is the aim of an autoencoder. MaxPooling, convolutional, and upsampling layers are used by autoencoders to denoise images.

For this experiment, the MNIST dataset is being used.
The handwritten numbers in the MNIST dataset are gathered together.

The assignment is to categorize a given image of a handwritten digit into one of ten classes, which collectively represent the integer values 0 through 9.

There are 60,000 handwritten, 28 X 28 digits in the dataset.
Here, a convolutional neural network is constructed. 

## Convolution Autoencoder Network Model

<img src=image-2.png width=350 height=90>

## DESIGN STEPS

### STEP 1:
Import Libraries

### STEP 2:
Load the dataset
### STEP 3:
Create a model
### STEP 4:
Compile the model and Display the images
### STEP 5:
End the program
## PROGRAM
### Name: SabariAkash A
### Register Number: 212222230124


```py
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_image = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(16, (3,3), activation="relu",padding='same')(input_image)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8, (3,3), activation="relu",padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8, (3,3), activation="relu",padding='same')(x)
encoder_output=layers.MaxPooling2D((2,2),padding='same')(x)



# Encoder output dimension is ## Mention the dimention ##

# Write your decoder here
x=layers.Conv2D(8, (3,3), activation="relu",padding='same')(encoder_output)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8, (3,3), activation="relu",padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16, (3,3), activation="relu")(x)
x=layers.UpSampling2D((2,2))(x)
decoder_output=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)


autoencoder1 = keras.Model(input_image, decoder_output)

print('Name:  SabariAKash A         Register Number: 212222230124       ')
autoencoder1.summary()

autoencoder1.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder1.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
metrics = pd.DataFrame(autoencoder1.history.history)
metrics.head()

print("sweety R")
metrics[['loss','val_loss']].plot()
decoded_imgs = autoencoder1.predict(x_test_noisy)

n = 10
print('Name: sweety R A\t Register Number: RA2211026010083       \n')
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img src=https://github.com/user-attachments/assets/323b97be-daf6-4d52-bbf8-37d5b2654312 width=250 height=200>

### Original vs Noisy Vs Reconstructed Image
<img src=image.png width=250 height=150>

## RESULT
Thus the Convolutional autoencoder for image denoising application is developed Successfully!
