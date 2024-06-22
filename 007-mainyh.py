import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

if __name__ == '__main__':
    train_dataSet = pd.read_csv('modified_数据集Time_Series448_detail.dat')
    test_dataSet = pd.read_csv('modified_数据集Time_Series660_detail.dat')

    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth', 'RECORD']
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth', 'Error_RECORD']

    X = train_dataSet[noise_columns].values
    y = train_dataSet[columns].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = X_train_tensor.shape[1]
    output_size = y_train_tensor.shape[1]
    model = NeuralNet(input_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    early_stopping = EarlyStopping(patience=10, delta=1e-6)

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch + 1}/50], Loss: {epoch_loss:.4f}')
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = model(X_val_tensor.to(device))
            val_loss = criterion(outputs, y_val_tensor.to(device)).item()

        print(f'Validation Loss: {val_loss:.4f}')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.eval()
    with torch.no_grad():
        X_val_tensor = X_val_tensor.to(device)
        y_val_pred = model(X_val_tensor).cpu().numpy()

    results = []
    for true_value, predicted_value in zip(y_val, y_val_pred):
        error = np.abs(true_value - predicted_value)
        formatted_true_value = ' '.join(map(str, true_value))
        formatted_predicted_value = ' '.join(map(str, predicted_value))
        formatted_error = ' '.join(map(str, error))
        results.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("result7.csv", index=False)

