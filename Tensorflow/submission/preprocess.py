import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def create_dataset_for_test(csv_path, batch_size=64, save_no_return=True):
     # Load the dataset
    df = pd.read_csv(csv_path)
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Preprocessing with sklearn (as an example)
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), ['Sex', 'Embarked']),
            ('scale', StandardScaler(), ['Age', 'SibSp', 'Parch', 'Fare'])
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(df)
    if save_no_return:
        save_preprocessed_data(X_processed)

    return X_processed

def save_preprocessed_data(X_processed, file_name="preprocessed_data.npy"):
    np.save(file_name, X_processed)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    create_dataset_for_test(csv_path)