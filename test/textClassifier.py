import torch.nn as nn
class TextClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # Single output neuron for binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # BCEWithLogitsLoss will apply sigmoid internally