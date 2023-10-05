import pytest
from project3 import * 

@pytest.fixture
def sample_data():
    '''
    Fixture to provide sample data for tests.
    Returns:
        pd.DataFrame: Sample data for testing purposes.
    '''
    return pd.DataFrame({
        'Age': [30, 40, 50],
        'Sex': ['M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ASY'],
        'HeartDisease': [0, 1, 1]
    })

def test_read_csv_file(sample_data):
    '''
    Test if the read_csv_file function correctly reads a CSV file.
    It saves sample data to a CSV file, then reads it and compares it with the original data.
    Args:
        sample_data (pd.DataFrame): Sample data for testing purposes.
    '''
    file_path = './files/heart_disease_prediction.csv'
    sample_data.to_csv(file_path, index=False)
    data = read_csv_file(file_path)
    assert data.equals(sample_data)

def test_display_dataframe(capsys, sample_data):
    '''
    Test if the display_dataframe function correctly displays DataFrame information.
    It captures the printed output and checks if specific column names are present.
    Args:
        capsys: Pytest fixture capturing stdout.
        sample_data (pd.DataFrame): Sample data for testing purposes.
    '''
    display_dataframe(sample_data)
    captured = capsys.readouterr()
    assert "Age" in captured.out

def test_column_types_and_counts(capsys, sample_data):
    '''
    Test if the column_types_and_counts function correctly displays column types and counts.
    It captures the printed output and checks if 'int64' and 'object' are present.
    Args:
        capsys: Pytest fixture capturing stdout.
        sample_data (pd.DataFrame): Sample data for testing purposes.
    '''
    column_types_and_counts(sample_data)
    captured = capsys.readouterr()
    assert "int64" in captured.out and "object" in captured.out

def test_show_unique_values(capsys, sample_data):
    '''
    Test if the show_unique_values function correctly displays unique values of specified columns.
    It captures the printed output and checks if the column name 'Sex' is present in the output.
    Args:
        capsys: Pytest fixture capturing stdout.
        sample_data (pd.DataFrame): Sample data for testing purposes.
    '''
    show_unique_values(['Sex', 'ChestPainType'], sample_data)
    captured = capsys.readouterr()
    assert "Column Sex:" in captured.out

def test_separate_features_label(sample_data):
    '''
    Test if the separate_features_label function correctly separates features and labels.
    It checks if the columns in features and labels match the expected columns.
    Args:
        sample_data (pd.DataFrame): Sample data for testing purposes.
    '''
    X, y = separate_features_label(sample_data)
    assert list(X.columns) == ['Age', 'Sex', 'ChestPainType']
    assert list(y) == [0, 1, 1]
