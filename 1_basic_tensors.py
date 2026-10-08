import torch
import numpy as np

# Multiple ways to init a tensor-
# 1. Directly from Data

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# 2. From a Numpy array-
np_array = np.array(data)
x_np = torch.from_numpy(np_array)  # not copying the data but both of them pointing towards the same memory location

# 3. From another tensor-
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor :\n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random Tensor: \n{x_rand}\n")

# 3. With Random or Constant values-
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n{rand_tensor}\n")
print(f"Ones Tensor: \n{ones_tensor}\n")
print(f"Zeros Tensor: \n{zeros_tensor}\n")

# Attributes of a tensor - Shape, Data Type and device
tensor = torch.rand(3,4)
print(f"Tensor shape: \n{tensor.shape}\n")
print(f"Tensor data type: \n{tensor.dtype}\n")
print(f"Tensor device: \n{tensor.device}\n")

# Check if hardware accelerators are present, if yes then switch from CPU to them using .to method-
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

print(f"{tensor.device}")  # now it will show cuda 0 instead of cpu

# Standard numpy like indexing and slicing-
tensor = torch.ones(4,4)
print(f"First Row: {tensor[0]}")
print(f"First Column: {tensor[:,0]}")
print(f"Last Column: {tensor[:,-1]}")
tensor[:, 1] = 0
print(tensor)

# Concatenate tensors using torch.cat-
t1 = torch.cat([tensor, tensor, tensor] , dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single element tensors-
agg = tensor.sum()   # adds all the individual elements inside tensor = 12
agg_item = agg.item()
print(agg_item , type(agg_item))  # this converts tensor wrapped value into Python standard numerical type(like float or int)


# In place operations-
print(tensor)
tensor.add_(5)
print(tensor)
# these save memory but for differentiation we need history so these are discouraged from that pov

# Bridge with NumPy-

# 1. Tensor to NumPy-
t = torch.ones(5)
print(t)
n = t.numpy()
print(n)

# now both t and n share the same memory and changing one will automatically change the other-
t.add_(5)
print(t)
print(n)


# Numpy array to Tensor-
n1 = np.ones(5)
t1 = torch.from_numpy(n1)

np.add(n1, 5, out = n1)
print(t1)
print(n1)