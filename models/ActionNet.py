from torch import nn


class EMG_LSTM(nn.Module):
    def __init__(self, num_classes, num_input):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=16, hidden_size=5, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=5, hidden_size=50, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(50, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        y = self.fc(y[:, -1, :])  # Take the output of the last time step
        return y, {}


class EMG_CNN(nn.Module):
    def __init__(self, num_classes, num_input):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.lstm = nn.LSTM(input_size=400, hidden_size=50, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(50, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        y = x.unsqueeze(1)
        y = self.conv(y)
        # Flatten the output for the LSTM
        y = y.view(y.size(0), y.size(1), -1)
        y, _ = self.lstm(y)
        y = self.fc(y[:, -1, :]) 
        return y, {}




class Early_Fusion(nn.Module):
    def __init__(self, num_classes, num_input):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=1040, hidden_size=5, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=5, hidden_size=50, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(50, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        y = self.fc(y[:, -1, :])  # Take the output of the last time step
        return y, {}