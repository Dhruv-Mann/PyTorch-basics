import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b  # standard linear transfomration --> y = xW + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # comparing model's prediction (z) with the actual labels (y)

# bbject in Function class stores all the operations performed on the tensor, it stores both forward and backward paths
print(f"Gradient function for z = {z.grad_fn}")   # the last operation performed was addition
print(f"Gradient function for loss = {loss.grad_fn}") # the last operation performed was the loss function

loss.backward()   # add retain_graph = True inside the paranthesis if you want to keep the computational graph as PyTorch throws it away after the very first run
print(w.grad)     # can only obtain grad for the leaf nodes of the computational graph which have requries_grad = True
print(b.grad)


# Disable Gradient Tracking
z = torch.matmul(x,w) + b
print(z.requires_grad)
with torch.no_grad():
    z= torch.matmul(x,w) + b
print(z.requires_grad)

# another way to achieve the same result- detach()
z = torch.matmul(x,w) + b
z_det = z.detach()
print(z_det.requires_grad)


# record of the entire data and operations is kept in DAG( directed acyclic graph) consisting of Function objects