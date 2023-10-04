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

def read_csv_file(file_path: str) -> pd.DataFrame:
    '''
    Reads a CSV file and returns its content as a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data 
        from the CSV file, or None if the file is not found.
    '''
    try:
        data = pd.read_csv(file_path)
        logging.info('✅ Data loaded successfully from file: %s', file_path)
        return data
    except FileNotFoundError:
        logging.error('❌ File not found at path: %s', file_path)
        return None

def display_dataframe(data: pd.DataFrame) -> None:
    '''
    Displays the first 5 rows of a Pandas DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to be displayed.

    Returns:
        None
    '''
    try:
        if isinstance(data, pd.DataFrame):
            display(data.head())
            display(data.dtypes.value_counts())
        else:
            raise ValueError('❌ Invalid input: Not a Pandas DataFrame')
    except ValueError as error:
        logging.error(error)
    except AttributeError:
        logging.error('❌ It was not possible to show the types of this file.')


def column_types_and_counts(data: pd.DataFrame) -> None:
    '''
    Displays the data types and counts of columns in a Pandas DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame for which column types are to be displayed.

    Returns:
        None
    '''
    try:
        if isinstance(data, pd.DataFrame):
            display(data.dtypes)
        else:
            raise ValueError('❌ Invalid input: Not a Pandas DataFrame')
    except ValueError as error:
        logging.error(error)

def show_description(data: pd.DataFrame, include: bool = False) -> None:
    '''
    Displays the description of a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to display the description of.
        include (bool, optional): Whether to include object data types. Defaults to False.

    Returns:
        None
    '''
    try:
        if include:
            description = data.describe(include=['object'])
        else:
            description = data.describe()
        display(description)
    except AttributeError as error:
        logging.error('❌ An error occurred while showing the description: %s', error)

def verify_na_values(data: pd.DataFrame) -> None:
    '''
    Verifies and displays missing values in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to be checked for missing values.

    Returns:
        None
    '''
    nan_counts = data.isna().sum()
    display(nan_counts)

    columns_with_nan = nan_counts[nan_counts != 0]
    if not columns_with_nan.empty:
        logging.warning("⚠️ There are columns with missing values. Details:")
        for column, count in columns_with_nan.items():
            logging.warning("Column '%s' has %s missing values.", column, count)
        logging.warning("Consider preprocessing these values soon!")
    else:
        logging.info("✅ No missing values found in the DataFrame.")

def show_unique_values(columns: list, data: pd.DataFrame) -> None:
    '''
    Displays unique values for specified columns in the DataFrame.

    Args:
        columns (list): List of column names to display unique values for.
        data (pd.DataFrame): The DataFrame to extract unique values from.

    Returns:
        None
    '''
    try:
        for column in columns:
            if column in data.columns:
                display(f'Column {column}:')
                display(data[column].unique())
            else:
                logging.warning("⚠️ Column '%s' does not exist in the dataframe.", column)
    except KeyError:
        logging.error("❌ One of the columns passed in the list doesn't exist in the dataframe")

def categorical_plot(cols_list: list, data: pd.DataFrame) -> None:
    '''
    Displays countplots for specified categorical columns in the DataFrame.

    Args:
        cols_list (list): List of column names to create countplots for.
        data (pd.DataFrame): The DataFrame containing the categorical columns.

    Returns:
        None
    '''
    try:
        # Computes the number of lines and columns in the plot
        cols = 2
        cols_size = len(cols_list)
        lines = cols_size // cols + (cols_size % cols > 0)

        # Creates the figure and subplots
        fig, axes = plt.subplots(lines, cols, figsize=(16, 15))
        axes = axes.flatten()

        for idx, col in enumerate(cols_list):
            if col in data.columns:
                sns.countplot(x=data[col], ax=axes[idx])
                axes[idx].set_title(f'Countplot for {col}')
                # Add data labels to each bar
                for container in axes[idx].containers:
                    axes[idx].bar_label(container, label_type="center")
            else:
                logging.warning("⚠️ Column '%s' does not exist in the dataframe.", col)

        # Remove empty subplots
        for i in range(len(cols_list), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    except KeyError:
        logging.error('''❌ One of the columns passed in the list
                            doesn't exist in the dataframe''')

def categorial_plot_by_label(cols_list: list, label: str, data: pd.DataFrame) -> None:
    '''
    Displays countplots for specified categorical columns in the DataFrame, 
    grouped by the specified label column.

    Args:
        cols_list (list): List of column names to create countplots for.
        label (str): The column name to group the countplots by.
        data (pd.DataFrame): The DataFrame containing the categorical columns and the label column.

    Returns:
        None
    '''
    try:
        # Computes the number of lines and columns in the plot
        cols = 2
        cols_size = len(cols_list)
        lines = cols_size // cols + (cols_size % cols > 0)

        # Creates the figure and subplots
        fig, axes = plt.subplots(lines, cols, figsize=(16, 15))
        axes = axes.flatten()

        for idx, col in enumerate(cols_list[:-1]):
            if col in data.columns:
                sns.countplot(x=data[col], hue=data[label], ax=axes[idx])
                axes[idx].set_title(f'Countplot for {col} Grouped by {label}')
                # Add data labels to each bar
                for container in axes[idx].containers:
                    axes[idx].bar_label(container, label_type="center")
            else:
                logging.warning("⚠️ Column '%s' does not exist in the dataframe.", col)

        # Remove empty subplots
        for i in range(len(cols_list), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    except KeyError:
        logging.error("❌ One of the columns passed in the list doesn't exist in the dataframe")

def show_zero_values(columns: list, data: pd.DataFrame) -> None:
    '''
    Identifies and displays rows where specified columns have values equal to zero.

    Args:
        columns (list): List of column names to check for zero values.
        data (pd.DataFrame): The DataFrame containing the specified columns.

    Returns:
        None
    '''
    try:
        for column in columns:
            zero_values = data[data[column] == 0]
            if not zero_values.empty:
                logging.info("Rows with zero values in column %s.", column)
                display(zero_values)
            else:
                logging.info("No zero values found in column %s.", column)
    except KeyError as error:
        logging.error("❌ Error: Column %s does not exist in the dataframe.", error.args[0])

def show_correlations(data: pd.DataFrame, limit: float = None) -> None:
    '''
    Calculates and plots the absolute correlations of a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame for which correlations need to be calculated.
        limit (float, optional): The correlation threshold for displaying correlations.
            Must be between 0 and 1. Defaults to None.

    Returns:
        None
    '''
    try:
        if limit is not None:
            if 0 < limit < 1:
                correlations = abs(data.corr())
                plt.figure(figsize=(12, 8))
                sns.heatmap(correlations[correlations > limit], annot=True, cmap="Blues")
            else:
                raise InvalidLimitCorrelationFilter('❌ The limit must be between 0 and 1!')
        else:
            correlations = abs(data.corr())
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlations, annot=True, cmap="Blues")
    except InvalidLimitCorrelationFilter as error:
        logging.error(error)

def separate_features_label(data: pd.DataFrame) -> tuple:
    '''
    Separates features from the label in the given DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing both features and labels.

    Returns:
        tuple: A tuple containing two elements - X (features) and y (labels).
    '''
    X = data.drop(columns=["HeartDisease"])
    y = data["HeartDisease"]
    return X, y

def train_knn(features: list, x_train: pd.DataFrame, y_train: pd.Series,
              x_val: pd.DataFrame, y_val: pd.Series) -> None:
    '''
    Trains KNN classifiers for each specified feature and calculates the accuracies.

    Args:
        features (list): List of feature names to train the classifiers.
        x_train (pd.DataFrame): DataFrame containing training features.
        y_train (pd.Series): Series containing training labels.
        x_val (pd.DataFrame): DataFrame containing validation features.
        y_val (pd.Series): Series containing validation labels.

    Returns:
        None
    '''
    for feature in features:
        # Defining the model
        knn = KNeighborsClassifier(n_neighbors=3)
        logging.info('ℹ️ Starting training with feature %s', feature)
        # Training the model
        knn.fit(x_train[[feature]], y_train)
        # Calculating the accuracy
        accuracy = knn.score(x_val[[feature]], y_val)
        accuracy_formatted = round(accuracy * 100, 2)
        logging.info('''The k-NN classifier trained on %s and
        with k = 3 has an accuracy of %s%%''', feature, accuracy_formatted)


def comparing_distributions(column: str, X: pd.DataFrame, x_train: pd.DataFrame,
                            x_val: pd.DataFrame) -> None:
    '''
    Compares the distributions of a column in the complete, training, and test datasets.

    Args:
        column (str): The column name to compare distributions.
        X (pd.DataFrame): The complete dataset.
        x_train (pd.DataFrame): The training dataset.
        x_val (pd.DataFrame): The validation dataset.

    Returns:
        None
    '''
    logging.info("Distribution of patients by their %s in the entire dataset", column)
    logging.info(X[column].value_counts())

    logging.info("\nDistribution of patients by their %s in the training dataset", column)
    logging.info(x_train[column].value_counts())

    logging.info("\nDistribution of patients by their %s in the test dataset", column)
    logging.info(x_val[column].value_counts())

def main():
    '''
    This is the function executed when we run the program
    '''

    # Read and explore the dataframe
    data = read_csv_file("./files/heart_disease_prediction.csv")
    display_dataframe(data)
    column_types_and_counts(data)
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
    display_dataframe(data_clean)
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
