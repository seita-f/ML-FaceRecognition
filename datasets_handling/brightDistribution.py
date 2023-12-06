import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_brightness(image):
    # Assuming image is in grayscale
    return np.mean(image)

def plot_brightness_distribution_with_outliers_boxplot(train_images, test_images, train_labels, test_labels, threshold=2.5):
    train_brightness = np.array([calculate_brightness(img) for img in train_images])
    test_brightness = np.array([calculate_brightness(img) for img in test_images])

    # Calculate z-scores to identify outliers
    z_scores_train = (train_brightness - np.mean(train_brightness)) / np.std(train_brightness)
    z_scores_test = (test_brightness - np.mean(test_brightness)) / np.std(test_brightness)

    # Identify outliers based on threshold
    outliers_train = np.abs(z_scores_train) > threshold
    outliers_test = np.abs(z_scores_test) > threshold

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot brightness distribution and boxplot for outliers
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Brightness Distribution
    sns.histplot(train_brightness, bins=50, color='blue', alpha=0.7, label='Train', ax=axs[0])
    sns.histplot(test_brightness, bins=50, color='orange', alpha=0.7, label='Test', ax=axs[0])
    axs[0].set_title('Brightness Distribution - Train and Test Datasets')
    axs[0].set_xlabel('Brightness')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Subplot 2: Boxplot for Outliers
    sns.boxplot(x=['Train']*len(train_brightness) + ['Test']*len(test_brightness),
                y=np.concatenate([train_brightness, test_brightness]), hue=['Train']*len(train_brightness) + ['Test']*len(test_brightness),
                ax=axs[1], palette={'Train': 'lightblue', 'Test': 'lightcoral'})
    sns.scatterplot(x=np.concatenate([np.ones_like(train_brightness[outliers_train]), 2 * np.ones_like(test_brightness[outliers_test])]),
                    y=np.concatenate([train_brightness[outliers_train], test_brightness[outliers_test]]),
                    hue=['Train Outliers']*sum(outliers_train) + ['Test Outliers']*sum(outliers_test),
                    marker='^', palette={'Train Outliers': 'red', 'Test Outliers': 'green'},
                    ax=axs[1])

    axs[1].set_title('Boxplot for Outliers - Train and Test Datasets')
    axs[1].set_ylabel('Brightness')
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(['Train', 'Test'])
    axs[1].legend()

    plt.tight_layout()
    plt.show()
