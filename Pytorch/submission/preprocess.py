import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_dataloader_for_test(csv_path, batch_size=64, save_no_return=True):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Drop columns that won't be used
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Only features since test data doesn't have 'Survived'
    X = df

    # Define a transformer for one-hot encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), ['Sex', 'Embarked']),
            ('scale', StandardScaler(), ['Age', 'SibSp', 'Parch', 'Fare'])
        ],
        remainder='passthrough'  # 'Pclass' can be left as-is since it's ordinal
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(X_processed, dtype=torch.float32)

    # Create TensorDataset without targets
    dataset = TensorDataset(features_tensor)

    # Create DataLoader
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data

    if save_no_return:
        # Save the DataLoader object
        torch.save(loader, 'dataloader.pth')

    return loader

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    create_dataloader_for_test(csv_path)