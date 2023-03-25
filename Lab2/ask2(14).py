import os
from glob import glob
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import cm
from pomegranate import *
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from sklearn.metrics import confusion_matrix
import itertools

def read_wav(f):
    wav, _ = librosa.core.load(f, sr=None)
    return wav

def parse_free_digits(directory):
    
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("\\")[1].split(".")[0].split("_") for f in files]
    
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]

    _, Fs = librosa.core.load(files[0], sr=None)

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers

def extract_features(wavs, n_mfcc=13, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames

def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train,train_ids = [], [], [],[]
    X_test, y_test, spk_test,test_ids = [], [], [],[]
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
            test_ids.append(idx)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)
            train_ids.append(idx)

    return X_train, X_test, y_train, y_test, spk_train, spk_test, train_ids,test_ids

def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale

def parser(directory, n_mfcc=13):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test,train_ids,test_ids = split_free_digits(frames, 
                                                    ids, speakers, y)

    return X_train, X_test, y_train, y_test, spk_train, spk_test,train_ids ,test_ids

def data_partition (wavs, ids, digits):
    list1 = []
    for i in range(len(ids)):
        list1.append(i)
    total_val_digits = 10*[0]
    max_val_digits = int(len(wavs)*0.2/10)
    training_wavs = []
    training_ids = []
    training_digits = []
    validation_wavs = []
    validation_ids = []
    validation_digits = []
    for i in range(len(ids)):
        index = random.sample(list1,1)
        index = index[0]
        list1.remove(index)
        digit = digits[index]
        total_val_digits[digit] += 1
        if total_val_digits[digit] <= max_val_digits:
            validation_wavs.append(wavs[index])
            validation_ids.append(ids[index])
            validation_digits.append(digit)
        else:
            training_wavs.append(wavs[index])
            training_ids.append(ids[index])
            training_digits.append(digit)
   
    return training_wavs, training_ids, training_digits, validation_wavs, validation_ids, validation_digits

def scale_data(X):
    scaler = StandardScaler()
    scaled_data=[]
    for i in range(len(X)):
        d = X[i]
        scaler.fit(d)
        h = scaler.transform(d)
        scaled_data.append(h)
    return scaled_data

class Data(Dataset):
    def __init__(self,features,labels):
        self.labels = labels
        self.lengths = []
        self.feature_dim=np.shape(features[0])[1]
        
        for i in range(len(labels)):
            s = np.shape(features[i])[0]
            self.lengths.append(s)
      
        self.max_seq_len = 153
        self.feats = self.zero_pad_and_stack(features)

        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')
    
    def zero_pad_and_stack(self,feat):
        padded=[]
        
        for i in range(len(self.labels)):
            temp = feat[i]
            extra = self.max_seq_len - self.lengths[i]
            temp = np.pad(temp,[(0,extra),(0,0)],)
            padded.append(temp)
            
        return padded


    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self,item):
        return self.feats[item],self.labels[item],self.lengths[item]
        
class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers,dropout, bidirectional,pps):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.rnn_size = rnn_size 
        self.pps = pps
        # Initialize the LSTM, Dropout, Output layers
        
        self.lstm = nn.LSTM(input_dim,rnn_size,num_layers,batch_first= True,
                            bidirectional = self.bidirectional)
        
        if(self.bidirectional == True):
            self.con = 2 
        else:
            self.con = 1
            
        self.fc = nn.Linear(self.con * rnn_size , output_dim)
        
    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """

        # You must have all of the outputs of the LSTM, but you 
        # need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        
        if(self.pps == True):
            out,_ = self.lstm(x)
            
        else:
            h0 = torch.zeros(self.con* self.num_layers,x.size(0),self.rnn_size)
            c0 = torch.zeros(self.con* self.num_layers,x.size(0),self.rnn_size)
            out, _ = self.lstm(x,(h0,c0))
        
        
        if(dropout == 1):
            drop = nn.Dropout(0.25)
            out = drop(out)
            
        out = self.fc(out)  
            
        last_outputs = self.last_timestep(out,lengths,self.bidirectional)

        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()
  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return  

##### MAIN  ######

xpr_train , xpr_test, y_train , y_test , spk_train, spk_test,train_ids,test_ids = parser("recordings",13)

x_train = scale_data(xpr_train)
x_test =  scale_data(xpr_test)

tr_wavs , tr_ids , tr_dig , val_wavs , val_ids , val_dig = data_partition(x_train,train_ids,y_train)

pps = False

if(pps == True):
    lens = []
    for i in range(len(tr_wavs)):
        lens.append(np.shape(tr_wavs[i])[0])
    
    zipped = list(zip(tr_wavs,tr_dig,lens))
    zipped.sort(key=lambda x:x[2],reverse=True)
    tr_wavs , tr_dig , lens = list(zip(*zipped))

EPOCHS = 30
BATCH_SZ = 20

if(pps == True):
    train_data = Data(tr_wavs,tr_dig)
    train_dl = DataLoader(train_data, batch_size = BATCH_SZ, shuffle = False)

else:
    train_data = Data(tr_wavs,tr_dig)
    train_dl = DataLoader(train_data, batch_size = BATCH_SZ, shuffle = True)

val_data = Data(val_wavs,val_dig)
val_dl = DataLoader(val_data, batch_size = BATCH_SZ , shuffle = False)

test_data = Data(xpr_test,y_test)
test_dl = DataLoader(test_data, batch_size = BATCH_SZ , shuffle = False)

input_dim = train_data.feature_dim
seq_length = train_data.max_seq_len
output_dim = len(unique_labels(tr_dig))
num_layers = 3
dropout = 0
bidirectional = True
rnn_size = 100

model = BasicLSTM(input_dim,rnn_size,output_dim,num_layers,dropout,bidirectional,pps)
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3  # the ETA variable in gradient descent

l2 = False
early_stopping_param = True
min_loss = np.Inf
noimprovement = 0
maxnoimprovement = 3*len(train_dl)  #--> we will wait for 2 epochs
                                    # and exit if loss has not improved 
stop_flag = False
path = 'torchsave/checkpoint.t7'

if(l2 == False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

model.train() # gradients "on"

start_time = time.time()
loss_per_epoch = []
val_loss_per_epoch=[]

for epoch in range(EPOCHS): # loop through dataset

    for i, data in enumerate(val_dl): # loop thorugh batches
       X_batch, y_batch, len_batch = data # get the features and labels
        
       optimizer.zero_grad() # ALWAYS USE THIS!! 
       out = model(X_batch,len_batch) # forward pass
       loss = criterion(out, y_batch) # compute per batch loss 
       val_loss_per_epoch.append(loss)
     
    running_average_loss = 0
    for i, data in enumerate(train_dl): # loop thorugh batches
    
        X_batch, y_batch, len_batch = data # get the features and labels
        optimizer.zero_grad() # ALWAYS USE THIS!! 
        out = model(X_batch,len_batch) # forward pass
        loss = criterion(out, y_batch) # compute per batch loss 
        loss.backward() # compure gradients based on the loss function
        optimizer.step() # update weights 
        running_average_loss += loss.detach().item()  
        
            
    print("Epoch: {} \t Loss {}".format(epoch, float(running_average_loss) / (i + 1)))
    loss_per_epoch.append((running_average_loss)/(i + 1))
    if(stop_flag==True):
        break

print("--- %s seconds ---" % (time.time() - start_time))

if(early_stopping_param == True):
    checkpoint = torch.load(path)    
    model.load_state_dict(checkpoint['state_dict'])   
    ep = checkpoint['epoch']       
    ba = checkpoint['batch']
    print("Model from Epoch: {} \t and Batch: {}".format(ep,ba))

model.eval() # turns off batchnorm/dropout ...
val_acc = 0
n_samples = 0
pred=[]

with torch.no_grad(): # no gradients required !! eval mode, speeds up computation
    for i, data in enumerate(val_dl):
        X_batch, y_batch,len_batch = data # test data and labels
        out = model(X_batch,len_batch) # get net's predictions
        val, y_pred = out.max(1) # argmax since output is a prob distribution
        pred.append(y_pred)
        val_acc += (y_batch == y_pred).sum().detach().item() # get accuracy
        n_samples += X_batch.size(0)

print('Validation Accuracy: ', 100 * val_acc / n_samples)

pred = np.concatenate(pred)
pr = list(pred)

cm = confusion_matrix(val_dig, pr)
plt.figure(1)
plot_confusion_matrix(cm, np.arange(10))


model.eval() # turns off batchnorm/dropout ...
test_acc = 0
n_samples = 0
pred=[]

with torch.no_grad(): # no gradients required !! eval mode, speeds up computation
    for i, data in enumerate(test_dl):
        X_batch, y_batch,len_batch = data # test data and labels
        out = model(X_batch,len_batch) # get net's predictions
        val, y_pred = out.max(1) # argmax since output is a prob distribution
        pred.append(y_pred)
        test_acc += (y_batch == y_pred).sum().detach().item() # get accuracy
        n_samples += X_batch.size(0)

print('Testing Accuracy: ', 100 * test_acc / n_samples)

pred = np.concatenate(pred)
pr = list(pred)

cmt = confusion_matrix(y_test, pr)
plt.figure(2)
plot_confusion_matrix(cmt, np.arange(10))