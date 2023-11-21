import numpy as np
import matplotlib.pyplot as plt

""" Before Image Arguementation """
def EDA(X, X_train, Y_age, Y_gender, Y_age_train, Y_gender_train):

    print("----- Exploratory data analysis -----")
    print(f'Original dataset size: {len(X)}')
    print(f'Train dataset size: {len(X_train)} \n {round(len(X_train)/len(X),2)*100}% of Original datasets')

    Y_age = np.array(Y_age, dtype=int)
    Y_gender = np.array(Y_gender, dtype=int)
    Y_age_train = np.array(Y_age, dtype=int)
    Y_gender_train = np.array(Y_gender, dtype=int)

    """ Image """
    # Check data with age
    plt.figure(figsize=(10, 10))
    for i in range(6):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i])
        plt.title(f"Age: {Y_age[i]}, Gender:{Y_gender[i]}")
        plt.axis('off')
    plt.show()

    fig = plt.figure(figsize=(15, 7))

    """ Age Disribution """
    # First subplot: Age Distribution in Original Data
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(Y_age, bins=20, edgecolor='black')
    ax1.set_title('Age Distribution in Original Data')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(range(0, max(Y_age)+1, 5))
    ax1.set_xticklabels(range(0, max(Y_age)+1, 5))

    # Second subplot: Age Distribution in Training Data
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(Y_age_train, bins=20, edgecolor='black')
    ax2.set_title('Age Distribution in Training Data')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks(range(0, max(Y_age_train)+1, 5))
    ax2.set_xticklabels(range(0, max(Y_age_train)+1, 5))

    # Display graph
    plt.show()


    """ Gender Disribution """
    # Create age bins
    age_bins = np.arange(0, max(max(Y_age), max(Y_age_train)) + 6, 5)

    # Count gender occurrences in each age bin
    male_counts, _ = np.histogram(Y_age[Y_gender == 0], bins=age_bins)
    female_counts, _ = np.histogram(Y_age[Y_gender == 1], bins=age_bins)

    male_counts_train, _ = np.histogram(Y_age_train[Y_gender_train == 0], bins=age_bins)
    female_counts_train, _ = np.histogram(Y_age_train[Y_gender_train == 1], bins=age_bins)

    # Calculate percentages
    total_counts = male_counts + female_counts
    total_counts_train = male_counts_train + female_counts_train

    male_percentages = male_counts / total_counts.astype(float)
    female_percentages = female_counts / total_counts.astype(float)

    male_percentages_train = male_counts_train / total_counts_train.astype(float)
    female_percentages_train = female_counts_train / total_counts_train.astype(float)

    # Plot stacked bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.bar(age_bins[:-1], male_percentages, color='blue', label='Male', width=5)
    ax1.bar(age_bins[:-1], female_percentages, bottom=male_percentages, color='pink', label='Female', width=5)

    ax1.set_title('Gender Distribution by Age in Original Data')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Percentage')
    ax1.legend()

    ax2.bar(age_bins[:-1], male_percentages_train, color='blue', label='Male', width=5)
    ax2.bar(age_bins[:-1], female_percentages_train, bottom=male_percentages_train, color='pink', label='Female', width=5)

    ax2.set_title('Gender Distribution by Age in Training Data')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Percentage')
    ax2.legend()

    plt.show()


# """ After Image Arguementation """
