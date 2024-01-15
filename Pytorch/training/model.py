import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_dataloader(csv_path, batch_size=64):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Drop columns that won't be used
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Separate features and target variable
    y = df['Survived']
    X = df.drop('Survived', axis=1)

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
    target_tensor = torch.tensor(y.values, dtype=torch.long)

    # Create TensorDataset
    dataset = TensorDataset(features_tensor, target_tensor)

    # Create DataLoader
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return loader

class TitanicSurvivalModel(nn.Module):
    def __init__(self):
        super(TitanicSurvivalModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(10, 14) 
        self.fc2 = nn.Linear(14, 7)
        self.output = nn.Linear(7, 2) # Binary classification (Survived or Not Survived)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

def train(model, data_loader, epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for inputs, labels in data_loader:
    
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        # Print loss every 10 epochs (or less frequently depending on your preference)
        if epoch % 10 == 9: 
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

        running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    # model_rep = torch.jit.script(TitanicSurvivalModel())
    # model_rep.save("untrained_model.pth")
    
    model = TitanicSurvivalModel()
    train_loader = create_dataloader('train.csv')
    train(model, train_loader)
    trained_model_rep = torch.jit.script(model)
    trained_model_rep.save("../submission/trained_model.pth")