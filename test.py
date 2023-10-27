import torch
import torch.nn as nn
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Create a trainable parameter using nn.Parameter
        #self.my_parameter = nn.Parameter(torch.Tensor(3, 3))  # Create a 3x3 parameter matrix
        self.my_parameter = nn.Parameter(torch.tensor(0.2))
        #nn.init.xavier_uniform_(self.my_parameter)  # Initialize the parameter

    def forward(self, x):
        # You can use self.my_parameter in the forward pass
        #out = x @ self.my_parameter
        out = x * self.my_parameter
        return out


# Create an instance of the model
model = MyModel()

# Move the model and its parameters to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create some input data (e.g., random data) and corresponding target data
input_data = torch.randn(3, 3).to(device)
target_data = torch.randn(3, 3).to(device)

# Define a loss function
criterion = nn.MSELoss()

# Create an optimizer (e.g., Stochastic Gradient Descent, SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
output = model(input_data)

# Compute the loss
loss = criterion(output, target_data)

# Zero the gradients to clear any previous gradients
optimizer.zero_grad()

# Backpropagation
loss.backward()

# Update the model's parameters
optimizer.step()

# Gradients have been computed, and model parameters have been updated

# You can access and modify the trainable parameter
print("My Parameter:")
print(model.my_parameter)
