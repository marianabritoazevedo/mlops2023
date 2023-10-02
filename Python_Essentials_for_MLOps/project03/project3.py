'''
This module aims to explore the KNN classifier to
predict heart diseases
'''

import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class InvalidLimitCorrelationFilter(Exception):
    '''
    Raised when a invalid limit is used to filter the correlation plot
    '''

def read_file(link):
    '''
    This function aims to read a csv file
    '''
    try:
        data = pd.read_csv(link)
        logging.info('✅ Data loaded successfully!')
        return data
    except FileNotFoundError:
        logging.error('❌ File not found.')
        return None

def show_file(data):
    '''
    This function aims to show the first 5 lines of a Pandas dataframe
    '''
    try:
        display(data.head())
    except AttributeError:
        logging.error('❌ It was not possible to show this file.')

def show_types_and_counts(data):
    '''
    This function aims to show the types of the columns and the quantity of each type in a dataframe
    '''
    try:
        display(data.dtypes)
        display(data.dtypes.value_counts())
    except AttributeError:
        logging.error('❌ It was not possible to show the types of this file.')

def show_description(data, include=False):
    '''
    This function aims to show the description of a dataframe
    '''
    try:
        if include:
            display(data.describe(include=['object']))
        else:
            display(data.describe())
    except AttributeError:
        logging.error('❌ It was not possible to show the description of this file.')

def verify_na_values(data):
    '''
    This function aims to verify if there are any missing values in the dataframe
    '''
    nan_counts = data.isna().sum()
    display(nan_counts)

    # Verify if there are Nan values in the columns
    columns_with_nan = nan_counts[nan_counts != 0]
    if not columns_with_nan.empty:
        logging.info("⚠️ There are some columns with some Nan values. They are:")
        for column, count in columns_with_nan.items():
            logging.info("Column %s has %s Nan values.", column, count)
        logging.info("You may have to preprocess these values soon!")

def show_unique_values(columns, data):
    '''
    This function aims to show the unique values for a list of columns of a dataframe
    '''
    try:
        for column in columns:
            display(f'Column {column}:')
            display(data[column].unique())
    except KeyError:
        logging.error("❌ One of the columns passed in the list doesn't exist in the dataframe")

def categorical_plot(cols_list, data):
    '''
    This function aims to show a countplot for categorical columns in a dataframe
    '''
    # Creates the figure
    fig = plt.figure(figsize=(16,15))

    # Computes the number of lines in the plot
    cols = 2
    cols_size = len(cols_list)
    lines = int(cols_size/2) if cols_size % 2 == 0 else int(cols_size/2)+1

    try:
        for idx, col in enumerate(cols_list):
            axis = plt.subplot(lines, cols, idx+1)
            sns.countplot(x=data[col], ax=axis)
            # Add data labels to each bar
            for container in axis.containers:
                axis.bar_label(container, label_type="center")
    except KeyError:
        logging.error('''❌ One of the columns passed in the list
                            doesn't exist in the dataframe''')

def categorial_plot_by_label(cols_list, label, data):
    '''
    This function aims to show a countplot for categorical columns in a dataframe, 
    making groups according to the label
    '''
    fig = plt.figure(figsize=(16,15))

    # Computes the number of lines in the plot
    cols = 2
    cols_size = len(cols_list)
    lines = int(cols_size/2) if cols_size % 2 == 0 else int(cols_size/2)+1

    try:
        for idx, col in enumerate(cols_list[:-1]):
            axis = plt.subplot(lines, cols, idx+1)
            # Group by the label of the dataframe
            sns.countplot(x=data[col], hue=data[label], ax=axis)
            # Add data labels to each bar
            for container in axis.containers:
                axis.bar_label(container, label_type="center")
    except KeyError:
        logging.error("❌ One of the columns passed in the list doesn't exist in the dataframe")

def show_zero_values(columns, data):
    '''
    This function is designed to identify and display, for a specific list of columns, 
    those that contain values equal to zero
    '''
    try:
        for column in columns:
            display(column)
            display(data[data[column] == 0])
    except KeyError:
        logging.error("❌ One of the columns passed in the list doesn't exist in the dataframe")

def show_correlations(data, limit=None):
    '''
    This function aims to calculate the correlations of a dataframe and plot them
    '''
    try:
        correlations = abs(data.corr())
        plt.figure(figsize=(12,8))
        if limit is not None:
            if limit > 0 or limit < 1:
                sns.heatmap(correlations[correlations > limit], annot=True, cmap="Blues")
            else:
                raise InvalidLimitCorrelationFilter('❌ The limit must be between 0 and 1!')
        else:
            sns.heatmap(correlations, annot=True, cmap="Blues")
    except InvalidLimitCorrelationFilter as error:
        print(error)

def separate_features_label(data):
    '''
    This function aims to separate features from the label
    '''
    X = data.drop(["HeartDisease"], axis=1)
    y = data["HeartDisease"]
    return X, y

def train_knn(features, x_train, y_train, x_val, y_val):
    '''
    This functions aims to build KNN classifiers with one feature and calculate the accuracies
    '''
    for feature in features:
        # Defining the model
        knn = KNeighborsClassifier(n_neighbors = 3)
        logging.info('ℹ️ Starting training with feature %s', feature)
        # Training the model
        knn.fit(x_train[[feature]], y_train)
        # Calculating the accuracy
        accuracy = knn.score(x_val[[feature]], y_val)
        accuracy_formated = round(accuracy*100, 2)
        logging.info('''The k-NN classifier trained on %s and with
                         k = 3 has an accuracy of %s''', feature, accuracy_formated)

def comparing_distributions(column, X, x_train, x_val):
    '''
    This function aims to compare the distributions of a column
    in the complete, training and test dataset
    '''

    print(f"Distribution of patients by their {column} in the entire dataset")
    print(X[column].value_counts())

    print(f"\nDistribution of patients by their {column} in the training dataset")
    print(x_train[column].value_counts())

    print(f"\nDistribution of patients by their {column} in the test dataset")
    print(x_val[column].value_counts())

def main():
    '''
    This is the function executed when we run the program
    '''

    # Read and explore the dataframe
    data = read_file("./files/heart_disease_prediction.csv")
    show_file(data)
    show_types_and_counts(data)
    show_description(data)

    # Verify Nan values and unique values
    verify_na_values(data)
    show_description(data, include=True)
    show_unique_values(['FastingBS', 'HeartDisease'], data)

    # EDA for categorical cols
    categorical_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG",
    "ExerciseAngina", "ST_Slope", "HeartDisease"]
    categorical_plot(categorical_cols, data)
    categorial_plot_by_label(categorical_cols, "HeartDisease", data)

    # Verify columns with the value 0
    columns = ["RestingBP", "Cholesterol"]
    show_zero_values(columns, data)

    # Creates a new copy to start data cleaning
    data_clean = data.copy()

    # Only keep non-zero values for RestingBP
    data_clean = data_clean[data_clean["RestingBP"] != 0]

    # Creates a mask according to the value of HeartDisease
    mask = data_clean["HeartDisease"]==0

    # Selects Cholesterol column according to the mask
    cholesterol_without_heartdisease = data_clean.loc[mask, "Cholesterol"]
    cholesterol_with_heartdisease = data_clean.loc[~mask, "Cholesterol"]

    # Replate values for Cholesterol according to the mask
    data_clean.loc[mask, "Cholesterol"] = cholesterol_without_heartdisease.replace(
        to_replace = 0, value = cholesterol_without_heartdisease.median())
    data_clean.loc[~mask, "Cholesterol"] = cholesterol_with_heartdisease.replace(
        to_replace = 0, value = cholesterol_with_heartdisease.median())
    show_description(data_clean)

    # Feature selection
    data_clean = pd.get_dummies(data_clean, drop_first=True)
    show_file(data_clean)
    show_correlations(data_clean)

    # Building KNN classifier for one feature
    X, y = separate_features_label(data_clean)
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.7, random_state=417)

    # Selecting features to apply in KNN
    features = ["Oldpeak", "Sex_M", "ExerciseAngina_Y", "ST_Slope_Flat", "ST_Slope_Up"]
    train_knn(features, x_train, y_train, x_val, y_val)

    # Building KNN classifier for multiple features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train[features])
    x_val_scaled = scaler.transform(x_val[features])

    # Trains the KNN model with all features of the dataset
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(x_train_scaled, y_train)
    accuracy = knn.score(x_val_scaled, y_val)
    print(f"Accuracy: {accuracy*100:.2f}")

    # Hyperparameter optimization
    grid_params = {"n_neighbors": range(1, 20),
                "metric": ["minkowski", "manhattan"]
              }

    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, grid_params, scoring='accuracy')
    knn_grid.fit(x_train_scaled, y_train)

    # Selecting the best parameters
    best_score = knn_grid.best_score_*100
    best_params = knn_grid.best_params_
    logging.info('The best parameters for KNN are: %s, %s', best_score, best_params)

    # Predictions and results on test set
    x_test_scaled = scaler.transform(x_val[features])
    predictions = knn_grid.best_estimator_.predict(x_test_scaled)
    accuracy = accuracy_score(y_val, predictions)
    print(f"Model Accuracy on test set: {accuracy*100:.2f}")

    # Comparing distributions
    comparing_distributions('Sex_M', X, x_train, x_val)

if __name__== "__main__":
    main()
