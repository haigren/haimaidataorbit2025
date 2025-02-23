import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import groq


# TODO: generate a GraphQL schema for each data set
# Generate a GraphQL schema from your dataset
schema = generate_schema_from_dataset(dataset)

# TODO: Use GROQ to query the schema and retrieve the data
query = """
  query {
    users(filter: { age: { gt: 18 } }) {
      id
      name
      email
    }
  }
"""
result = query(schema, query)

# TODO: Convert the result from GROQ data into a PyTorch dataset
# Convert the result to a PyTorch dataset
class UserDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx]
        return {
            'id': user['id'],
            'name': user['name'],
            'email': user['email']
        }

dataset = UserDataset(result.data.users) 


# TODO: Create a PyTorch DataLoader from the dataset
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# TODO: Define a PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # input layer (3) -> hidden layer (128)
        self.fc2 = nn.Linear(128, 10)  # hidden layer (128) -> output layer (10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

model = Net()

# TODO Later:  Define a loss function and optimizer (probably for accuracy)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# TODO: Train the model
for epoch in range(10):
    for batch in data_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#model will predict if text is a hate speech or not, if it is then send it to groq to rephrase it, else leave it as it is
#want to catagorize if it's a hate speech and notify user 