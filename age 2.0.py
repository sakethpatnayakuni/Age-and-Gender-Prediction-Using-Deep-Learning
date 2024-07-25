#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)
num_classes = 8  # Number of age categories

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'C:\Users\saket\OneDrive\Documents\age&gender2.0\final documentation\classified' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labeling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  # Assume folder names are age categories, e.g., "0-10", "11-20", etc.
    if label == "0-12":
        label = 0
    elif label == "13-25":
        label = 1
    elif label == "26-35":
        label = 2
    elif label == "36-45":
        label = 3
    elif label == "46-55":
        label = 4
    elif label == "56-65":
        label = 5
    elif label == "66-75":
        label = 6
    else:  # "76+"
        label = 7
        
    labels.append([label])  # [[0], [1], [2], ...]

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=num_classes)  # [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], ...]
testY = to_categorical(testY, num_classes=num_classes)

# augmenting dataset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# define model
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":  # Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1
    
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))  # Use softmax for multi-class classification

    return model

# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=num_classes)

# define optimizer
opt = Adam(learning_rate=lr)

# compile model
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs, verbose=1
)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Age Classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("age_classification_plot.png")


# In[3]:


from tensorflow.keras.models import load_model

# Load the trained model
model.save('age_detection.h5')


# In[5]:


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load the age detection model
model = load_model('age_detection.h5')

# Define age categories
age_categories = ['0-12', '13-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']

# Open webcam
webcam = cv2.VideoCapture(0)

# Loop through frames
while webcam.isOpened():

    # Read frame from webcam 
    status, frame = webcam.read()

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):

        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for age detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply age detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix, e.g., [[0.1, 0.2, ..., 0.1]]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = age_categories[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Age Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()


# In[ ]:




