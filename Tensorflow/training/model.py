import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Function to create a DataLoader
def create_dataset(csv_path, batch_size=64):
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

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=len(X_train))

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset

# Define the model using TensorFlow/Keras
class TitanicSurvivalModel(tf.keras.Model):
    def __init__(self):
        super(TitanicSurvivalModel, self).__init__()
        # Define layers
        self.fc1 = tf.keras.layers.Dense(14, activation='relu', input_shape=(10,))
        self.fc3 = tf.keras.layers.Dense(7, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc3(x)
        x = self.output_layer(x)
        return x

# Function to train the model
def train(model, train_data, val_data, epochs=500, batch_size=64):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        for batch_x, batch_y in train_data:
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss_value = loss_fn(batch_y, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss += loss_value

        # Validation loop
        for batch_x, batch_y in val_data:
            logits = model(batch_x, training=False)
            loss_value = loss_fn(batch_y, logits)
            val_loss += loss_value

        # Print loss every 10 epochs (or less frequently depending on your preference)
        if epoch % 10 == 9:
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss / len(train_data)}, Validation Loss: {val_loss / len(val_data)}')

    print('Finished Training')

if __name__ == "__main__":
    train_data, val_data = create_dataset('train.csv')
    
    model = TitanicSurvivalModel()
    
    train(model, train_data, val_data)
    
    # Save the trained model
    model.save("../submission/trained_model.h5")