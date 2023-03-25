from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np

class Data(Dataset):
    def __init__(self, X, y, trans=None):
        self.data = list(zip(X, y))
        self.trans = trans
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.trans is not None:
            return self.trans(self.data[index])
        else:
            return self.data[index]
        
class LinearWActivation(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(LinearWActivation, self).__init__()
        self.f = nn.Linear(in_features, out_features)
        if activation == 'sigmoid':
            self.a = nn.Sigmoid()
        else:
            self.a = nn.ReLU()
            
    def forward(self, x): 
        return self.a(self.f(x))
    
class FullyConnectedNN(nn.Module): 
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
        super(FullyConnectedNN, self).__init__()
        layers_in = [n_features] + layers 
        layers_out = layers + [n_classes]
        self.f = nn.Sequential(*[
          LinearWActivation(in_feats, out_feats, activation=activation)
          for in_feats, out_feats in zip(layers_in, layers_out)
      ])
        self.clf = nn.Linear(n_classes, n_classes)
        
    def forward(self, x):
        y = self.f(x)
        return self.clf(y)
        
    
mytest = np.loadtxt("test.txt")
mytrain = np.loadtxt("train.txt")

y_train = mytrain[:,0]
y_test = mytest[:,0]

 

x_train = mytrain[:,1:257]
x_test = mytest[:,1:257]

 

train_data = Data(x_train, y_train)
myNN = FullyConnectedNN([100,100], x_train.shape[1], 10)

 

print(f"The network architecture is: \n {myNN}")
        
        
        

        
        
    
        
    


 