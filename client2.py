import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import flwr as fl
from sklearn.model_selection import train_test_split
import datetime
import random

X = np.load('X2.npy')
Y = np.load('Y2.npy')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

#X = pd.concat([pd.DataFrame(X_polyp), pd.DataFrame(X_non_polyp)],axis = 0)
#Y = pd.concat([pd.DataFrame(y_polyp), pd.DataFrame(y_non_polyp)],axis = 0)
'''X = np.vstack((X_polyp, X_non_polyp))
Y = np.vstack((y_polyp, y_non_polyp))
combined = np.hstack((X,Y))
np.random.shuffle(combined)
split_index = int(0.8*len(combined))
X_train = combined[:split_index, :-1]  # All rows until split_index, all columns except the last one
y_train = combined[:split_index, -1]   # All rows until split_index, only the last column
X_test = combined[split_index:, :-1]   # All rows from split_index onwards, all columns except the last one
y_test = combined[split_index:, -1]    # All rows from split_index onwards, only the last column

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

combined2 = np.hstack((X_train, y_train))
np.random.shuffle(combined2)
split_index = int(0.8*len(combined2))
X_train = combined2[:split_index, :-1]  # All rows until split_index, all columns except the last one
y_train = combined2[:split_index, -1]   # All rows until split_index, only the last column
X_val = combined2[split_index:, :-1]   # All rows from split_index onwards, all columns except the last one
y_val = combined2[split_index:, -1]    # All rows from split_index onwards, only the last column

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)'''

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation and testing sets (only rescaling)
val_test_datagen = ImageDataGenerator()

# Train data generator
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32
)

# Validation data generator
val_generator = val_test_datagen.flow(
    X_val,
    y_val,
    batch_size=32
)

# Test data generator
test_generator = val_test_datagen.flow(
    X_test,
    y_test,
    batch_size=32,
    shuffle=False
)

# Load MobileNet model (pre-trained on ImageNet)
mobilenet_model = tf.keras.applications.MobileNet(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pre-trained layers
for layer in mobilenet_model.layers:
    layer.trainable = False

# Add classification head
model = tf.keras.models.Sequential([
    mobilenet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
print("Test Accuracy:", test_accuracy)
'''
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        weights = model.get_weights()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = random.randint(0, 100)
        # model.save(f"R1C2-{current_time}-{random_suffix}_MODEL.h5")
        return weights

    def fit(self, parameters, config):
        model.set_weights(parameters)
        weights = model.get_weights()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = random.randint(0, 100)
        # model.save(f"R1C2.1-{current_time}-{random_suffix}_MODEL.h5")
        np.save(f"PreClient2-{current_time}-{random_suffix}_weights.npy", model.get_weights())
        for i in range(5):
            model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // 32,
                validation_data=val_generator,
                validation_steps=len(X_val) // 32
            )
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = random.randint(0, 100)
        # model.save(f"Client2-{current_time}-{random_suffix}_model.h5")
        np.save(f"PostClient2-{current_time}-{random_suffix}_weights.npy", model.get_weights())
        return weights, len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
        return test_loss, len(X_test), {"accuracy": test_accuracy}


fl.client.start_client(server_address="127.0.0.1:5000", client=FlowerClient().to_client())
