# First, import PyTorch
import torch
def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable
features = torch.randn((1, 5))
# Features are 5 random normal variables
v = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

print(activation(torch.sum(torch.mm(features,weights.view(5,1)))))
