from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes, num_input):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

        self.model = nn.Sequential(
            # Input shape: (num_input, 1024)
            nn.Conv1d(in_channels=num_input, out_channels=num_input*2, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(num_input*2),
            nn.ReLU(),
            # Shape: (num_input*2, 256)
            nn.Conv1d(in_channels=num_input*2, out_channels=num_input*4, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(num_input*4),
            nn.ReLU(),
            # Shape: (num_input*4, 64)
            nn.Conv1d(in_channels=num_input*4, out_channels=num_input*8, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(num_input*8),
            nn.ReLU(),
            # Shape: (num_input*8, 16)
            nn.Conv1d(in_channels=num_input*8, out_channels=num_classes, kernel_size=16, stride=1, padding=0),
            # Shape: (num_classes, 1)
            nn.Sigmoid()
        )

        

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y = self.model(x)
        return y.squeeze(), {}
        return self.model(x).squeeze(), {}
    

class LSTM(nn.Module):
    def __init__(self, num_classes, num_input):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

        

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y, _ = self.lstm(x)
        y = self.fc(y[:, -1, :])  # Take the output of the last time step
        return y, {}
    

class MLP(nn.Module):
    def __init__(self, num_classes, num_input):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(num_input*1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

        

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.model(x)
        return y, {}