import pytest
from project3 import * 

# Fixture para fornecer dados de amostra para os testes
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Age': [30, 40, 50],
        'Sex': ['M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ASY'],
        'HeartDisease': [0, 1, 1]
    })

# Testa se a função read_csv_file lê corretamente um arquivo CSV
def test_read_csv_file(sample_data):
    file_path = './files/heart_disease_prediction.csv'
    sample_data.to_csv(file_path, index=False)
    data = read_csv_file(file_path)
    assert data.equals(sample_data)

# Testa se a função display_dataframe exibe corretamente informações sobre o DataFrame
def test_display_dataframe(capsys, sample_data):
    display_dataframe(sample_data)
    captured = capsys.readouterr()
    assert "Age" in captured.out

# Testa se a função column_types_and_counts exibe corretamente os tipos e contagens de colunas
def test_column_types_and_counts(capsys, sample_data):
    column_types_and_counts(sample_data)
    captured = capsys.readouterr()
    assert "int64" in captured.out and "object" in captured.out

# Testa se a função show_unique_values exibe corretamente os valores únicos das colunas especificadas
def test_show_unique_values(capsys, sample_data):
    show_unique_values(['Sex', 'ChestPainType'], sample_data)
    captured = capsys.readouterr()
    assert "Column Sex:" in captured.out

# Testa se a função separate_features_label separa corretamente as características e rótulos
def test_separate_features_label(sample_data):
    X, y = separate_features_label(sample_data)
    assert list(X.columns) == ['Age', 'Sex', 'ChestPainType']
    assert list(y) == [0, 1, 1]
