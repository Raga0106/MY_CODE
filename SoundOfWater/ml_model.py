# ml_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WaterLevelPredictor(nn.Module):
    def __init__(self):
        super(WaterLevelPredictor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

        # Define model architecture, increased neurons and layers
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.to(self.device)  # Move model to device

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict(self, acoustic_data):
        features = np.array([[acoustic_data['frequency'], acoustic_data['amplitude']]], dtype=np.float32)
        features = StandardScaler().fit_transform(features)  # Standardize features
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).to(self.device)
            prediction = self.forward(features_tensor)
        return prediction.item()

    def train_model(self, training_data):
        self.train()  # Set model to training mode
        min_samples_for_training = 100

        if len(training_data) < min_samples_for_training:
            logging.warning(
                f"Not enough samples for training. Current: {len(training_data)}, Required: {min_samples_for_training}")
            return False

        X = np.array([[d['frequency'], d['amplitude']] for d in training_data], dtype=np.float32)
        y = np.array([d['level'] for d in training_data], dtype=np.float32)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Standardize features

        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).view(-1, 1).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        num_epochs = 100  # Increased number of epochs
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        return True

    def evaluate(self, X_test, y_test):
        self.eval()  # Set model to evaluation mode
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test.astype(np.float32))  # Standardize test set
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).to(self.device)
            y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).to(self.device)

            predictions = self.forward(X_test_tensor).cpu().numpy()
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            logging.info(f"Model evaluation - MSE: {mse}, R-squared: {r2}")
        return mse, r2
