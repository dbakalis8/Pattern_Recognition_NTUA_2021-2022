import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random

class Data(Dataset):
    def __init__(self,X,y,transform=None):
        self.data = list(zip(X,y))
        self.trans = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]


class myRNN(nn.Module):
    def __init__(self, in_features,out_features,hidden_size,num_layers):
        super(myRNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(in_features,hidden_size,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size,out_features)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out, _ = self.rnn(x,h0)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

class myLSTM(nn.Module):
    def __init__(self, in_features,out_features,hidden_size,num_layers):
        super(myLSTM,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_features,hidden_size,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size,out_features)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        
        out, _ = self.lstm(x,(h0,c0))
        out = out[:,-1,:]
        out = self.fc(out)
        return out

class myGRU(nn.Module):
    def __init__(self, in_features,out_features,hidden_size,num_layers):
        super(myGRU,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(in_features,hidden_size,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size,out_features)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out, _ = self.gru(x,h0)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

    
#### INITIALIZATIONS ####
f=40
reps = 100
samples = 10
t = np.zeros((reps,samples))
start=0
end = 1/f

for i in range(reps):
    k = np.random.uniform(low=start,high=end,size=1).astype(float)
    k = float(k)
    a = np.linspace(k,1/f+k,10)
    t[i,:]=a
    
t = np.sort(t)
val_sin = torch.sin(2*np.pi*f*torch.FloatTensor(t))
val_cos = torch.cos(2*np.pi*f*torch.FloatTensor(t))

num = random.randint(0,reps-1)
asin = val_sin[num]
acos = val_cos[num]

plt.figure(0)
plt.plot(t[num],asin.numpy(),label ="Sin wave")
plt.plot(t[num],acos.numpy(),label ="Cos wave")
plt.xlabel("Time axis")
plt.ylabel("Amplitude")
plt.title("Sin/Cos of Randdom sample: {}".format(num))
plt.grid()
plt.legend()
plt.show()

epochs = 3
batch_size = 10
hidden_size = 100
input_size = 10
output_size = 10
num_layers = 2

X_train, X_test, y_train, y_test = train_test_split(val_sin, val_cos, test_size=0.1)
train_data = Data(X_train,y_train,None)
test_data =  Data(X_test,y_test,None)

train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = myRNN(input_size,output_size,hidden_size,num_layers)

criterion = nn.MSELoss()
learning_rate = 1e-2  # the ETA variable in gradient descent
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs): # loop through dataset
    running_average_loss = 0
    for i, data in enumerate(train_dl): # loop thorugh batches
        X_batch, y_batch = data # get the features and labels
        optimizer.zero_grad() # ALWAYS USE THIS!! 
        X_batch = X_batch.reshape(-1,1,10)
        out = model(X_batch) # forward pass
        loss = criterion(out, y_batch) # compute per batch loss 
        loss.backward() # compurte gradients based on the loss function
        optimizer.step() # update weights 
        
        running_average_loss += loss.detach().item()
        if i % 100 == 0:
            print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))

     
model.eval() # turns off batchnorm/dropout ...
acc = 0
n_samples = 0
with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
    for i, data in enumerate(test_dl):
        X_batch, y_batch = data # test data and labels
        X_batch = X_batch.reshape(-1,1,10)
        out = model(X_batch) # get net's predictions
        val, y_pred = out.max(1) # argmax since output is a prob distribution
        acc += (y_batch == y_pred).sum().detach().item() # get accuracy
        n_samples += X_batch.size(0)
        

fig = plt.figure(1)

for i in range(5):
    for j in range(2):
        fig.add_subplot(5,2,2*i+j+1)
        plt.plot(t[0],y_batch[2*i+j].numpy(),label='cosine wave')
        plt.plot(t[0],out[2*i+j],label ='cosine prediction' )
        plt.grid()
        plt.legend()
plt.show() 