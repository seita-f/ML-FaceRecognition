""" Module """
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
import os, zipfile, io, re
# from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

from datasets_handling.EDA import *
from datasets_handling.imageArgumentation import *
from datasets_handling.brightDistribution import *
from datasets_handling.brightDistribution_age import *
from datasets_handling.contrastDistribution import *


""" PATH """
# path for input dir
path = 'input'

# path for datasets dir
# dataset_path = ['datasets/our_dataset']
dataset_path  = ['datasets/part3']
# dataset_path  = ['datasets/part1', 'datasets/part2', 'datasets/part3']

""" Handling argument (reading in the data) """
# if len(sys.argv) != 2:
#     print("Usage: python main.py path/to/image.jpg")
#     sys.exit(1)
#
# image_path = sys.argv[1]
# image_filename = os.path.basename(image_path)
#
# image = cv2.imread(image_path)

# cv2.imwrite(os.path.join(path, image_filename), image)
# cv2.waitKey(0)

""" initial dataset collected """
# Set your image size
image_size = 100

# Function to load images from a folder
def load_images_from_folder(dataset_path):
    images = []
    labels_age = []
    labels_gender = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(dataset_path, filename)

            try:
                # Read the image
                image = cv2.imread(file_path)

                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Resize the image
                    image = cv2.resize(image, (image_size, image_size))

                    # Append the image to the list
                    images.append(image)


                    # [age]_[gender]_[race]_[date&time].jpg
                    label_parts = filename.split('_')

                    # Get age and gender
                    label_age = label_parts[0]
                    label_gender = label_parts[1]

                    labels_age.append(label_age)
                    labels_gender.append(label_gender)

                # else:
                #     print(f"Skipping {file_path} - Unable to read image.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(images), np.array(labels_age), np.array(labels_gender)


# Initialize empty lists for data and labels
X = []
Y_age = []
Y_gender = []

# Load images and labels from each folder
for folder_path in dataset_path:
    images, labels_age, labels_gender = load_images_from_folder(folder_path)
    X.extend(images)
    Y_age.extend(labels_age)
    Y_gender.extend(labels_gender)


""" Handling datasets """
X = np.array(X)
Y_age = np.array(Y_age)
Y_gender = np.array(Y_gender)

# Shuffle data
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
Y_age = Y_age[rand_index]
Y_gender = Y_gender[rand_index]

# Split this into train and test part
X_train = X[:int(len(X)*0.8)]  # 80%
Y_age_train = Y_age[:int(len(Y_age)*0.8)]  # 80%
Y_gender_train = Y_gender[:int(len(Y_gender)*0.8)]  # 80%

X_test = X[int(len(X)*0.8):]   # 20%
Y_age_test = Y_age[int(len(Y_age)*0.8):]   # 20%
Y_gender_test = Y_gender[int(len(Y_gender)*0.8):]   # 20%

# normalize the data (0-255 => 0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


""" Exploratory data analysis """
EDA(X, X_train, Y_age, Y_gender, Y_age_train, Y_gender_train)
plot_brightness_distribution_with_outliers_boxplot(X_train, X_test, Y_age_train, Y_age_test)
plot_brightness_distribution_with_outliers_kde(X_train, X_test, Y_age_train, Y_age_test, 2.5, 20)
plot_contrast_distribution_with_outliers_kde(X_train, X_test, Y_age_train, Y_age_test, 2.5, 20)

""" Image Arguementation """
imageArgumentation(X_train, Y_age_train)

""" Exploratory data analysis after Image Arguementation """
