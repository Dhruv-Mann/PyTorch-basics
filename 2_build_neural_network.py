import os
import torch
from torch import nn
from torch.utils.data import dataloader
from torchvision import datasets, transforms

# hardware accelerator check-
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Defining neural network by subclassing nn.Module-
class NeuralNetwork(nn.Module):     # subclassing nn.Module
    def __init__(self):     
        # DEFINING THE LAYERS IN INIT        
        super().__init__()          # calls the constructor of nn.Module to init all internal PyTorch machinery before we define our own custom layers
        self.flatten = nn.Flatten() # flatten the 2D image (28x28 pixels) into flat 1D vector (784 pixels)
        self.linear_relu_stack = nn.Sequential(  # chains the layers together so that the data flows through them in order
            nn.Linear(784, 512),  # takes the 784 pixels, multiply them by a weight matrix and convert them into 512 features
            nn.ReLU(),            # converts any negative values into 0 to add non linearity
            nn.Linear(512,512),   # 512 more features
            nn.ReLU(),
            nn.Linear(512,10)     # Output Layers - maps the 512 features onto 10 final raw output scores
        )
    def forward(self, x):   # specifies how input data x flows through the layers defined inside __init__
        # DEFINING THE PIPELINES IN FORWARD
        x = self.flatten(x)  # 1. x gets flattened
        logits = self.linear_relu_stack(x) # 2.x passes through linear_relu_stack
        return logits  # 3. it returns logits—unnormalized, raw output values for each class.

model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1, 28 , 28 , device= device)  # generates a 3d tensor representing 1 image of 28x28 pixels filled with random values, device = cuda 0
logits = model(X)  # passes X through the Neural Network and returns a 2D tensor of shape (1,10)
pred_probab = nn.Softmax(dim=1)(logits) # Softmax scales all 10 raw logit scores into clean probability percentages between 0.0 and 1.0 that sum up to 1.0 (100%)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# Model layers-
input_image = torch.rand(3,28,28)
print(input_image.size())

# 1. nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 2. nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


# 3. nn.ReLU  (Rectified Linear Unit)  Formula: f(x) = max(0, x)   --> return either positive value or 0
print(f"Before ReLU: \n{hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: \n{hidden1}\n")

# 4 nn.Sequential
seq_models = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_models(input_image)

# 5. nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model Parameters
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")