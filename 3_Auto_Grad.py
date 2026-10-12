import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b  # standard linear transformation --> y = xW + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # comparing model's prediction (z) with the actual labels (y)

# Object in the Function class stores all the operations performed on the tensor; it stores both forward and backward paths.
print(f"Gradient function for z = {z.grad_fn}")   # the last operation performed was addition
print(f"Gradient function for loss = {loss.grad_fn}") # the last operation performed was the loss function

loss.backward()   # add retain_graph = True inside the parentheses if you want to keep the computational graph, as PyTorch throws it away after the first run
print(w.grad)     # can only obtain grad for the leaf nodes of the computational graph which have requires_grad = True
print(b.grad)


# Disable Gradient Tracking
z = torch.matmul(x,w) + b
print(z.requires_grad)
with torch.no_grad():
    z= torch.matmul(x,w) + b
print(z.requires_grad)

# Another way to achieve the same result - detach()
z = torch.matmul(x,w) + b
z_det = z.detach()
print(z_det.requires_grad)


# A record of the entire data and operations is kept in a DAG (directed acyclic graph) consisting of Function objects.