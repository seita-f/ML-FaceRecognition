import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # Import OpenCV

def calculate_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_image)

def split_data_by_age(images, labels, age_threshold):
    # Convert labels to integers (assuming labels are numeric)
    labels = labels.astype(int)

    # Use boolean indexing to split data based on age threshold
    younger_than_threshold = labels < age_threshold
    older_than_threshold = labels >= age_threshold

    # Split data into two groups
    younger_images = images[younger_than_threshold]
    older_images = images[older_than_threshold]

    younger_labels = labels[younger_than_threshold]
    older_labels = labels[older_than_threshold]

    return younger_images, older_images, younger_labels, older_labels

def plot_brightness_distribution_with_outliers_kde(train_images, test_images, train_labels, test_labels, threshold=2.5, age_threshold=40):
    # Split data by age
    train_images_younger, train_images_older, train_labels_younger, train_labels_older = split_data_by_age(train_images, train_labels, age_threshold)
    test_images_younger, test_images_older, test_labels_younger, test_labels_older = split_data_by_age(test_images, test_labels, age_threshold)

    # Calculate brightness for each group
    train_brightness_younger = np.array([calculate_brightness(img) for img in train_images_younger])
    train_brightness_older = np.array([calculate_brightness(img) for img in train_images_older])
    test_brightness_younger = np.array([calculate_brightness(img) for img in test_images_younger])
    test_brightness_older = np.array([calculate_brightness(img) for img in test_images_older])

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Brightness Distribution with KDE for Train Data
    sns.kdeplot(train_brightness_younger, color='blue', fill=True, label='Younger', ax=axs[0])
    sns.kdeplot(train_brightness_older, color='orange', fill=True, label='Older', ax=axs[0])
    axs[0].set_title('Density Plot of Average Brightness - Train Data')
    axs[0].set_xlabel('Average Brightness')
    axs[0].set_ylabel('Density')
    axs[0].legend()

    # Subplot 2: Brightness Distribution with KDE for Test Data
    sns.kdeplot(test_brightness_younger, color='blue', fill=True, label='Younger', ax=axs[1])
    sns.kdeplot(test_brightness_older, color='orange', fill=True, label='Older', ax=axs[1])
    axs[1].set_title('Density Plot of Average Brightness - Test Data')
    axs[1].set_xlabel('Average Brightness')
    axs[1].set_ylabel('Density')
    axs[1].legend()

    plt.show()
