
# Αλέξιος   Μάρας    03118074
# Δημήτριος Μπακάλης 03118163

## Βήμα 0 - Εξοικείωση με Kaggle kernels

# In[ ]:

import numpy as np
import pandas as pd
import os
import librosa.display as display
import matplotlib.pyplot as plt

#for dirname, _, filenames in os.walk('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/test'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
      
files = os.listdir("../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/")
print(files)
#print(np.load('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/test/3431.fused.full.npy'))


# # Bήμα 1 - Εξοικείωση με φασματογραφήματα στην κλίμακα mel
# 

## Βήμα 1(α)

# In[ ]:

path = "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt"
with open(path) as f:
    files = f.readlines()
    
files.pop(0)

label = ''
counter = 0
ids = 2*[0]

for i in range(len(files)):
    files[i] = files[i].split()
    if label != files[i][1] and counter < 2:
        print(files[i])
        label = files[i][1]
        ids[counter] = i
        counter += 1


## Βήμα 1 (β)

# In[ ]:

spectrogram_file = "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/"

f = os.listdir(spectrogram_file)
new_ids = 2*[0]
for i in range(len(f)):
    if (f[i] == files[ids[0]][0][:-3]):
        new_ids[0] = i
    elif (f[i] == files[ids[1]][0][:-3]):
        new_ids[1] = i


## Βήμα 1 (γ) 

# In[ ]:

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

spectrograms1 = np.load(spectrogram_file + f[new_ids[0]])
mel_spectrogram1 = spectrograms1[:128]
chromagram1 = spectrograms1[128:]
display.specshow(mel_spectrogram1, x_axis = 'time', y_axis = 'mel', ax = ax[0])
ax[0].set(title = 'Spectrogram of a sample with label: {}'.format(files[ids[0]][1]))
ax[0].label_outer()

spectrograms2 = np.load(spectrogram_file + f[new_ids[1]])
mel_spectrogram2 = spectrograms2[:128]
chromagram2 = spectrograms2[128:]
display.specshow(mel_spectrogram2, x_axis = 'time', y_axis = 'mel', ax = ax[1])
ax[1].set(title = 'Spectrogram of a sample with label: {}'.format(files[ids[1]][1]))
ax[1].label_outer()

print(mel_spectrogram1.shape, mel_spectrogram2.shape)


# # Βήμα 2 - Συγχρονισμός φασματογραφημάτων στο ρυθμό της μουσικής (beat-synced spectrograms)
# 

# In[ ]:

spec_beat_file = "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/"

f = os.listdir(spec_beat_file)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

beat_spectrograms1 = np.load(spec_beat_file + f[new_ids[0]])
beat_mel_spectrogram1 = beat_spectrograms1[:128]
beat_chromagram1 = beat_spectrograms1[128:]
display.specshow(beat_mel_spectrogram1, x_axis = 'time', y_axis = 'mel', ax = ax[0])
ax[0].set(title = 'Beat-synced spectrogram of a sample with label: {}'.format(files[ids[0]][1]))
ax[0].label_outer()

beat_spectrograms2 = np.load(spec_beat_file + f[new_ids[1]])
beat_mel_spectrogram2 = beat_spectrograms2[:128]
beat_chromagram2 = beat_spectrograms2[128:]
display.specshow(beat_mel_spectrogram2, x_axis = 'time', y_axis = 'mel', ax = ax[1])
ax[1].set(title = 'Beat-synced spectrogram of a sample with label: {}'.format(files[ids[1]][1]))
ax[1].label_outer()

print(beat_mel_spectrogram1.shape, beat_mel_spectrogram2.shape)


## Βήμα 3 - Εξοικείωση με χρωμογραφήματα

# In[ ]:

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)

display.specshow(chromagram1, x_axis = 'time', y_axis = 'chroma', ax = ax[0][0])
ax[0][0].set(title = 'Chromagram (left) and Beat-synced chromagram (right) of a sample with label: {}'.format(files[ids[0]][1]))
ax[0][0].label_outer()

display.specshow(beat_chromagram1, x_axis = 'time', y_axis = 'chroma', ax = ax[0][1])

display.specshow(chromagram2, x_axis = 'time', y_axis = 'chroma', ax = ax[1][0])
ax[1][0].set(title = 'Chromagram (left) and Beat-synced chromagram (right) of a sample with label: {}'.format(files[ids[1]][1]))
ax[1][0].label_outer()

display.specshow(beat_chromagram2, x_axis = 'time', y_axis = 'chroma', ax = ax[1][1])

print(chromagram1.shape, chromagram2.shape)
print(beat_chromagram1.shape, beat_chromagram2.shape)


## Βήμα 4 - Φόρτωση και ανάλυση δεδομένων

# In[ ]:

import copy
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


def torch_train_val_split(dataset, batch_train, batch_eval,
                          val_size=0.2, shuffle=True, seed=420):
    
    # Creating data indices for training and validation splits:
        
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
        
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    return train_loader, val_loader

def read_spectrogram(spectrogram_file,option, chroma=True):
    # with open(spectrogram_file, "r") as f:
    if(option == 'sp_only'):
        spectrograms = np.load(spectrogram_file)[:128]
        
    elif(option == 'chr_only'):
        spectrograms = np.load(spectrogram_file)[128:]
    
    else:   
        spectrograms = np.load(spectrogram_file)
    
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    
    return spectrograms.T

class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1
        
class SpectrogramDataset(Dataset):
    def __init__(self, path, option, class_mapping, 
                 train=True, max_length=-1, regression=None):
        
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression

        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f),option) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        
        if isinstance(labels, (list, tuple)):
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    dataset = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        class_mapping=CLASS_MAPPING, train=True,option = 'both')

    print(dataset[10])
    print(f"Input: {dataset[10][0].shape}")
    print(f"Label: {dataset[10][1]}")
    print(f"Original length: {dataset[10][2]}")
    print(len(dataset))
    
classes_before = ["Rock", "Psych-Rock", "Indie-Rock", "Post-Rock",
                  "Psych-Folk", "Folk", "Metal", "Punk", "Post-Punk", 
                  "Trip-Hop", "Pop", "Electronic", "Hip-Hop", "Classical", 
                  "Blues", "Chiptune", "Jazz", "Soundtrack", "International", "Old-Time"]

classes_after = ["Rock", "Folk", "Metal", "Trip-Hop", "Pop", 
                 "Electronic", "Hip-Hop", "Classical", "Blues", "Jazz"]


# In[ ]:

before_class_maping = []
after_class_maping = []

for file in files:
    for i in range (20):
        if file[1] == classes_before[i]:
            before_class_maping.append(classes_before[i])
            break
            
    if i == 0 or i == 1 or i == 3:
        after_class_maping.append("Rock")
    elif i == 4 or i == 5:
        after_class_maping.append("Folk")
    elif i == 6 or i == 7:
        after_class_maping.append("Metal")
    elif i == 9:
        after_class_maping.append("Trip-Hop")
    elif i == 10:
        after_class_maping.append("Pop")
    elif i == 11 or i == 15:
        after_class_maping.append("Electronic")
    elif i == 12:
        after_class_maping.append("Hip-Hop")
    elif i == 13:
        after_class_maping.append("Classical")
    elif i == 14:
        after_class_maping.append("Blues")
    elif i == 16:
        after_class_maping.append("Jazz")

 
from collections import Counter
import pandas
letter_counts1 = Counter(before_class_maping)
df = pandas.DataFrame.from_dict(letter_counts1, orient='index')
df.plot(kind='bar')

letter_counts2 = Counter(after_class_maping)
df = pandas.DataFrame.from_dict(letter_counts2, orient='index')
df.plot(kind='bar')


# # Βήμα 5 - Αναγνώριση μουσικού είδους με LSTM
# 

## Δημιουργία του LSTM μοντέλου

# In[ ]:

import torch.nn as nn
import torch
import pickle
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


# In[ ]:

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers,dropout,bidirectional):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.rnn_size = rnn_size 
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
        
        x = x.double()
        h0 = torch.zeros(self.con* self.num_layers,x.size(0),self.rnn_size).to(device)
        c0 = torch.zeros(self.con* self.num_layers,x.size(0),self.rnn_size).to(device)
        h0 = h0.double()
        c0 = c0.double()
        out, _ = self.lstm(x,(h0,c0))
        
        if(self.dropout == 1):
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


## Εκπαίδευση του LSTM

# In[ ]:

def train_lstm(train_loader,val_loader,output_dim,input_size,lf,filename,overfit_batch=False):
    
    input_dim = input_size
    num_layers = 3
    rnn_size = 100
    dropout = 0
    bidirectional = True
    patience = 10
    
    if(lf == 1):
        criterion = nn.CrossEntropyLoss()
    
    elif (lf == 0):
        criterion = nn.MSELoss()
    
    model = BasicLSTM(input_dim,rnn_size,output_dim,num_layers,dropout,bidirectional).to(device)
    learning_rate = 1e-4  # the ETA variable in gradient descent
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_per_epoch = []
    val_loss_per_epoch = []
    
    model.train()
    model = model.double()
    
    if(overfit_batch == False):
        EPOCHS = 50
        for epoch in range(EPOCHS):
            val_loss = 0
            for (i,data) in enumerate(val_loader):
                X_batch, y_batch, len_batch = data # get the features and labels
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
                out = model(X_batch,len_batch).double() # forward pass
                if(lf == 0 ):
                    new_shape = (len(y_batch), 1)
                    y_batch = y_batch.view(new_shape)
                loss = criterion(out, y_batch) # compute per batch loss 
                val_loss +=loss.detach().item()
            
            val_loss_per_epoch.append(val_loss/i)
            i = np.argmin(val_loss_per_epoch)
            if (i == epoch):
                best_model = model   
            if (epoch > i + patience):
                val_loss_per_epoch.pop(-1)
                print('Early stopping...')
                break
                
            running_average_loss = 0
            for (i,data) in enumerate(train_loader):
                X_batch, y_batch, len_batch = data # get the features and labels
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
            
                optimizer.zero_grad() # ALWAYS USE THIS!! 
                out = model(X_batch,len_batch).double() # forward pass
                
                if( lf == 0 ):
                    new_shape = (len(y_batch), 1)
                    y_batch = y_batch.view(new_shape)
                
                loss = criterion(out, y_batch) # compute per batch loss 
                loss.backward() # compure gradients based on the loss function
                optimizer.step() # update weights 
                l = loss.detach().item()
                running_average_loss += l
                
                if i % 10 == 0:
                    print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i,
                                        float(l)))
                
            running_average_loss = running_average_loss/(i+1)
            loss_per_epoch.append(running_average_loss)
    
    elif (overfit_batch == True):
        EPOCHS = 750
        X_batch , y_batch , len_batch = next(iter(train_loader))
        for epoch in range(EPOCHS):
            running_average_loss = 0
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            len_batch = len_batch.to(device)
            
            optimizer.zero_grad() # ALWAYS USE THIS!! 
            out = model(X_batch,len_batch).double() # forward pass
            loss = criterion(out, y_batch) # compute per batch loss 
            loss.backward() # compure gradients based on the loss function
            optimizer.step() # update weights 
            l = loss.detach().item()
            running_average_loss += l
            print("Epoch: {} \t Loss {}".format(epoch,
                                      float(l)))
                
            running_average_loss = running_average_loss
            loss_per_epoch.append(running_average_loss)
            
    pickle.dump(best_model, open(filename, 'wb'))    
    return loss_per_epoch,val_loss_per_epoch


## Plotting Validation Loss and Loss per Epoch

# In[ ]:

def plot_loss(loss_per_epoch,val_loss_per_epoch, overfit_batch):

    epochs = np.arange(0,len(loss_per_epoch),1)
    
    if(overfit_batch == False):
        plt.plot(epochs,loss_per_epoch,label = 'training loss')
        plt.plot(epochs,val_loss_per_epoch,label = 'validation loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.show()
    
    else:
        plt.plot(epochs,loss_per_epoch,label = 'training loss')
        plt.grid()
        plt.title('Training loss when overfitting with a Batch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.show()
    
    return


## Αξιολόγηση του μοντέλου

# In[ ]:

def test_model(test_loader,filename):
    model = pickle.load(open(filename, 'rb'))
    model.eval()
    test_acc = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for (i,data) in enumerate(test_loader):
            X_batch, y_batch, len_batch = data # get the features and labels
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            len_batch = len_batch.to(device)
            
            out = model(X_batch,len_batch).double()
            val,y_pred = out.max(1)
            
            for pred in y_pred:
                predictions.append(pred.item())
            for lab in y_batch:
                labels.append(lab.item())
                
    print(classification_report(labels,predictions))
    
    return


# In[ ]:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:

sp = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        class_mapping = CLASS_MAPPING, train = True, option = 'sp_only')
train_sp , val_sp = torch_train_val_split(sp, 32, 32, val_size=.3)


## Batch Overfit

# In[ ]:

tr_loss,val_loss = train_lstm(train_sp,val_sp,input_size = 128,output_dim = 10, lf = 1,
                              filename='lstm_sp.sav',overfit_batch = True)


# In[ ]:

plot_loss(tr_loss,val_loss,overfit_batch=True)


## Εκπαίδευση με το Φασματογράφημα

# In[ ]:

tr_loss,val_loss = train_lstm(train_sp,val_sp,input_size = 128, output_dim = 10, lf = 1,
                              filename='lstm_sp.sav',overfit_batch= False)


# In[ ]:

plot_loss(tr_loss,val_loss,overfit_batch=False)


# In[ ]:

sp_test = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/",
        class_mapping=CLASS_MAPPING, train = False ,option = 'sp_only')

sp_test_loader = DataLoader(sp_test,batch_size = 20 , shuffle = True )


# In[ ]:

test_model(sp_test_loader,filename = 'lstm_sp.sav')


## Εκπαίδευση για το Beat Synced Spectograms

# In[ ]:

bssp = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat",
        class_mapping=CLASS_MAPPING, train=True, option = 'sp_only')
train_bssp , val_bssp = torch_train_val_split(bssp, 32, 32, val_size=.3)


# In[ ]:

bssp_tr_loss,bssp_val_loss = train_lstm(train_bssp,val_bssp,input_size = 128,output_dim = 10, lf = 1,
                                        filename = 'lstm_bssp.sav',overfit_batch=False)


# In[ ]:

plot_loss(bssp_tr_loss,bssp_val_loss,overfit_batch = False)


# In[ ]:

bssp_test = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat",
        class_mapping=CLASS_MAPPING, train = False ,option = 'sp_only')

bssp_test_loader = DataLoader(bssp_test, batch_size = 20 , shuffle = True )


# In[ ]:

test_model(sp_test_loader,filename = 'lstm_bssp.sav')


## Eκπαίδευση για τα Χρωμογραφήματα

# In[ ]:

chr_data = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        class_mapping=CLASS_MAPPING, train = True, option = 'chr_only')

train_chr , val_chr = torch_train_val_split(chr_data, 32, 32, val_size=.3)


# In[ ]:

chr_tr_loss, chr_val_loss = train_lstm(train_chr , val_chr , input_size = 12, output_dim = 10, 
                                       lf = 1, filename = 'lstm_chr.sav' ,overfit_batch=False)


# In[ ]:

plot_loss(chr_tr_loss,chr_val_loss,overfit_batch=False)


# In[ ]:

print(chr_val_loss[-1])


# In[ ]:

chr_test = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        class_mapping=CLASS_MAPPING, train = False, option = 'chr_only')

chr_test_loader = DataLoader(chr_test,batch_size = 20 , shuffle = True)


# In[ ]:

test_model(chr_test_loader,filename = 'lstm_chr.sav')


## Εκπαίδευση για τα ενωμένα Χρωμογραφήματα - Φασματογραφήματα

# In[ ]:

train_spch , val_spch =  torch_train_val_split(dataset, 32, 32, val_size=.3)


# In[ ]:

spch_tr_loss, spch_val_loss = train_lstm(train_spch,val_spch,input_size = 140,output_dim = 10,
                                         lf = 1, filename = 'lstm_spch.sav', overfit_batch=False)


# In[ ]:

print(spch_val_loss[-1])


# In[ ]:

plot_loss(spch_tr_loss,spch_val_loss,overfit_batch = False)


# In[ ]:

spch_test = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        class_mapping=CLASS_MAPPING, train = False , option = 'both')

spch_test_loader = DataLoader(spch_test, batch_size = 20 , shuffle = True)


# In[ ]:

test_model(spch_test_loader,filename = 'lstm_spch.sav')


## Bήμα 7ο: CNN Model για ταξινόμηση του FMA

# In[ ]:

import torch.nn as nn
import torch.nn.functional as F

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*159*14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, lengths):
        x = torch.unsqueeze(x,1).double().to('cuda')
        
        x = self.conv1(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv2(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv3(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')

        x = torch.flatten(x, 1).to('cuda') # flatten all dimensions except batch
        x = F.relu(self.fc1(x)).to('cuda')
        x = F.relu(self.fc2(x)).to('cuda')
        x = self.fc3(x).to('cuda')
        
        return x


# In[ ]:

def train_cnn(model,train_loader,val_loader,lf,filename,overfit_batch=False):
    patience = 10
    learning_rate = 1e-4  
    
    if(lf == 1):
        criterion = nn.CrossEntropyLoss()
    
    elif (lf == 0):
        criterion = nn.MSELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_per_epoch = []
    val_loss_per_epoch = []
    
    model.train()
    model = model.double()
    
    if(overfit_batch == False):
        EPOCHS = 50
        for epoch in range(EPOCHS):
            val_loss = 0
            for (i,data) in enumerate(val_loader):
            
                X_batch, y_batch, len_batch = data # get the features and labels
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
                out = model(X_batch,len_batch).double() # forward pass
                
                if(lf == 0 ):
                    new_shape = (len(y_batch), 1)
                    y_batch = y_batch.view(new_shape)
                
                loss = criterion(out, y_batch) # compute per batch loss 
                val_loss +=loss.detach().item()
                
            val_loss_per_epoch.append(val_loss/i)
            i = np.argmin(val_loss_per_epoch)
            if (i == epoch):
                best_model = model     
            if (epoch > i + patience):
                val_loss_per_epoch.pop(-1)
                print('Early stopping...')
                break
            
            running_average_loss = 0
            for (i,data) in enumerate(train_loader):
                X_batch, y_batch, len_batch = data # get the features and labels
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
            
                optimizer.zero_grad() # ALWAYS USE THIS!! 
                out = model(X_batch,len_batch).double() # forward pass
                
                if(lf == 0 ):
                    new_shape = (len(y_batch), 1)
                    y_batch = y_batch.view(new_shape)
                    
                loss = criterion(out, y_batch) # compute per batch loss 
                loss.backward() # compure gradients based on the loss function
                optimizer.step() # update weights 
                l = loss.detach().item()
                running_average_loss += l
                if i % 10 == 0:
                    print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i,
                                        float(l)))
                
            running_average_loss = running_average_loss/(i+1)
            loss_per_epoch.append(running_average_loss)
    
    elif (overfit_batch == True):
        EPOCHS = 750
        X_batch , y_batch , len_batch = next(iter(train_loader))
        for epoch in range(EPOCHS):
            running_average_loss = 0
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            len_batch = len_batch.to(device)
            
            optimizer.zero_grad() # ALWAYS USE THIS!! 
            out = model(X_batch,len_batch).double() # forward pass
            loss = criterion(out, y_batch) # compute per batch loss 
            loss.backward() # compure gradients based on the loss function
            optimizer.step() # update weights 
            l = loss.detach().item()
            running_average_loss += l
            print("Epoch: {} \t Loss {}".format(epoch,
                                      float(l)))
                
            running_average_loss = running_average_loss
            loss_per_epoch.append(running_average_loss)
            
    pickle.dump(best_model, open(filename, 'wb'))  
    
    return loss_per_epoch,val_loss_per_epoch


## Overfit Batch με το CNN

# In[ ]:

model = CNN_model().to(device)

tr_loss,val_loss = train_cnn(model,train_sp,val_sp, lf = 1 ,
                             filename='cnn_sp.sav',overfit_batch = True)


# In[ ]:

plot_loss(tr_loss,val_loss,overfit_batch = True)


## Εκπαίδευση του CNN με τα φασματογραφήματα

# In[ ]:

model = CNN_model().to(device)
trc_loss,valc_loss = train_cnn(model,train_sp,val_sp, lf = 1,
                            filename='cnn_sp.sav',overfit_batch = False)


# In[ ]:

plot_loss(trc_loss,valc_loss,overfit_batch = False)


# In[ ]:

test_model(sp_test_loader,filename = 'cnn_sp.sav')


## Εκπαίδευση για τα ενωμένα Χρωμογραφήματα - Φασματογραφήματα

# In[ ]:

model = CNN_model().to(device)
spch_tr_loss, spch_val_loss = train_cnn(model,train_spch,val_spch, lf = 1,
                            filename = 'cnn_spch.sav',overfit_batch = False)


# In[ ]:

plot_loss(spch_tr_loss, spch_val_loss,overfit_batch = False)


# In[ ]:

test_model(spch_test_loader,filename = 'cnn_spch.sav')


## Βήμα 8ο : Εκτίμηση συναισθήματος με παλινδρόμηση

# In[ ]:

valence_labels = []
energy_labels = []
danceability_labels = []
ids = []

path = "../input/patreco3-multitask-affective-music/data/multitask_dataset/train_labels.txt"

with open(path) as f:
    files = f.readlines()
    
files.pop(0)

dataset = []
for i in range(len(files)):
    dataset.append(files[i].split(','))
    dataset[i][0] = int(dataset[i][0])
    dataset[i][1] = float(dataset[i][1])
    dataset[i][2] = float(dataset[i][2])
    dataset[i][3] = float(dataset[i][3][:-1])
    
for i in range(len(dataset)):
    ids.append(dataset[i][0])
    valence_labels.append(dataset[i][1])
    energy_labels.append(dataset[i][2])
    danceability_labels.append(dataset[i][3])
    
ids = np.array(ids)
valence_labels = np.array(valence_labels)
energy_labels = np.array(energy_labels)
danceability_labels = np.array(danceability_labels)

spectrogram_file = "../input/patreco3-multitask-affective-music/data/multitask_dataset/train/"
f = os.listdir(spectrogram_file)
zeros = np.zeros((140, 1))
specs = []
for i in range(len(f)):
    spec = np.load(spectrogram_file + f[i])
    id1 = int(f[i].split('.')[0])
    original_shape = spec.shape[1]
    if spec.shape[1] != 1293:
        while(spec.shape[1] < 1293):
            spec = np.concatenate((spec, zeros), axis=1)
    specs.append([spec.T, id1, original_shape])
    
specs_valence = []
specs_energy = []
specs_danceability = []
specs_multi=[]

for i in range(len(ids)):
    for spec in specs:
        if spec[1] == ids[i]:
            specs_valence.append(spec)
            specs_valence[-1][1] = valence_labels[i]
            specs_energy.append(spec)
            specs_energy[-1][1] = energy_labels[i]
            specs_danceability.append(spec)
            specs_danceability[-1][1] = danceability_labels[i]
            specs_multi.append(spec)
           # specs_multi[-1][1]= (valence_labels[i],energy_labels[i],danceability_labels[i])
            break


## CNN δίκτυο για παλινδρόμηση (Regression)

# In[ ]:

class CNN_model_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 159 * 15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x, lengths):
        x = torch.unsqueeze(x,1).double().to('cuda')
        
        x = self.conv1(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv2(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv3(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')

        x = torch.flatten(x, 1).to('cuda')# flatten all dimensions except batch
        x = F.relu(self.fc1(x)).to('cuda')
        x = F.relu(self.fc2(x)).to('cuda')
        x = self.fc3(x).to('cuda')
        
        return x


# In[ ]:

from scipy import stats

def model_eval_regression(val_loader,filename):
    model = pickle.load(open(filename, 'rb'))
    model.eval()
    pred = []
    true = []
    total_preds = 0
    with torch.no_grad(): 
        for i, data in enumerate(val_loader):
            x_batch, y_batch,len_batch = data 
            total_preds += len(len_batch)
            x_batch = x_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            len_batch = len_batch.to('cuda')
            out = model(x_batch,len_batch).double().to('cuda')
            y_pred, val = out.max(1)
            for j in range(len(y_pred)):
                pred.append(y_pred[j])
                true.append(y_batch[j])
        
    true = torch.Tensor(true)
    pred = torch.Tensor(pred)
    corr, pval = stats.spearmanr(true, pred)
    return corr


## Valence

# In[ ]:

valence_train_loader,valence_val_loader = torch_train_val_split(specs_valence,
                                                    32 ,32, val_size=.30)


## LSTM

# In[ ]:

lstm_valence_tr_loss,lstm_valence_val_loss = train_lstm(valence_train_loader,valence_val_loader,
                              input_size = 140, output_dim = 1, lf = 0 ,
                              filename = 'lstm_valence.sav',overfit_batch= False)


# In[ ]:

plot_loss(lstm_valence_tr_loss,lstm_valence_val_loss,overfit_batch = False)


# In[ ]:

corr = model_eval_regression(valence_val_loader,filename = 'lstm_valence.sav')
print('Spearman Correlation -> {}'.format(corr))


## CNN

# In[ ]:

cnn_regr = CNN_model_regression().to('cuda')

cnn_valence_tr_loss,cnn_valence_val_loss = train_cnn(cnn_regr,valence_train_loader,
                                                     valence_val_loader, lf = 0,
                                                     filename='cnn_valence.sav',overfit_batch = False)


# In[ ]:

plot_loss(cnn_valence_tr_loss,cnn_valence_val_loss,overfit_batch = False)


# In[ ]:

corr = model_eval_regression(valence_val_loader,filename = 'cnn_valence.sav')
print('Spearman Correlation -> {}'.format(corr))


## Energy

# In[ ]:

energy_train_loader, energy_val_loader = torch_train_val_split(specs_energy,
                                                    32 ,32, val_size=.33)


## LSTM

# In[ ]:

lstm_energy_tr_loss, lstm_energy_val_loss = train_lstm(energy_train_loader,energy_val_loader,
                              input_size = 140, output_dim = 1, lf = 0,
                              filename = 'lstm_energy.sav',overfit_batch= False)


# In[ ]:

plot_loss(lstm_energy_tr_loss, lstm_energy_val_loss,overfit_batch = False)


# In[ ]:

corr = model_eval_regression(energy_val_loader,filename = 'lstm_energy.sav')
print('Spearman Correlation -> {}'.format(corr))


## CNN

# In[ ]:

cnn_regr = CNN_model_regression().to('cuda')

cnn_energy_tr_loss, cnn_energy_val_loss = train_cnn(cnn_regr,energy_train_loader,
                                                     energy_val_loader, lf = 0,
                                                 filename = 'cnn_energy.sav',overfit_batch = False)


# In[ ]:

plot_loss(cnn_energy_tr_loss, cnn_energy_val_loss,overfit_batch = False)


# In[ ]:

corr = model_eval_regression(energy_val_loader,filename = 'cnn_energy.sav')
print('Spearman Correlation -> {}'.format(corr))


## Danceability

# In[ ]:

dance_train_loader, dance_val_loader = torch_train_val_split(specs_danceability,
                                                    32 ,32, val_size=.33)


## LSTM

# In[ ]:

lstm_dance_tr_loss, lstm_dance_val_loss = train_lstm(dance_train_loader,dance_val_loader,
                              input_size = 140, output_dim = 1, lf = 0,
                              filename = 'lstm_dance.sav',overfit_batch= False)


# In[ ]:

plot_loss(lstm_dance_tr_loss, lstm_dance_val_loss,overfit_batch = False)


# In[ ]:

corr = model_eval_regression(dance_val_loader,filename = 'lstm_dance.sav')
print('Spearman Correlation -> {}'.format(corr))


## CNN

# In[ ]:

cnn_regr = CNN_model_regression().to('cuda')

cnn_dance_tr_loss, cnn_dance_val_loss = train_cnn(cnn_regr,dance_train_loader,
                                                     dance_val_loader, lf = 0,
                                                 filename = 'cnn_dance.sav',overfit_batch = False)


# In[ ]:

plot_loss(cnn_dance_tr_loss, cnn_dance_val_loss,overfit_batch = False)


# In[ ]:

corr = model_eval_regression(dance_val_loader,filename = 'cnn_dance.sav')
print('Spearman Correlation -> {}'.format(corr))


## Βήμα 9ο : Transfer Learning

# In[ ]:

class CNN_for_Transfer_Learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 159 * 15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

 

    def forward(self, x, lengths):
        x = torch.unsqueeze(x,1).double().to('cuda')
        
        x = self.conv1(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv2(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv3(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')

 

        x = torch.flatten(x, 1).to('cuda') # flatten all dimensions except batch
        x = F.relu(self.fc1(x)).to('cuda')
        x = F.relu(self.fc2(x)).to('cuda')
        x = self.fc3(x).to('cuda')
        
        return x


## Training CNN on FMA dataset

# In[ ]:

cnn = CNN_for_Transfer_Learning().to(device)

 

tr_loss,val_loss = train_cnn(cnn,train_spch,val_spch, lf = 1 ,
                             filename='cnn_valence.sav',overfit_batch = False)

 

plot_loss(tr_loss,val_loss,overfit_batch = False)


## Saving Model and Retraining it in the new dataset

# In[ ]:

cnn = pickle.load(open('cnn_valence.sav', 'rb'))
cnn.fc3 = nn.Linear(84, 1)
cnn = cnn.to('cuda')
spch_tr_loss,spch_val_loss = train_cnn(cnn,valence_train_loader,valence_val_loader, lf = 0 ,
                             filename='cnn_valence.sav',overfit_batch = False)


# In[ ]:

plot_loss(spch_tr_loss, spch_val_loss,overfit_batch = False)
corr = model_eval_regression(valence_val_loader,filename = 'cnn_valence.sav')
print('Spearman Correlation -> {}'.format(corr))


## Βήμα 10ο: Εκπαίδευση σε Πολλαπλά Προβλήματα

## CNN Network for Multitask Learning

# In[ ]:

import torch.nn as nn
import torch.nn.functional as F

class CNN_multitask(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 159 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x, lengths):
        x = torch.unsqueeze(x,1).double().to('cuda')
        
        x = self.conv1(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv2(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')
        
        x = self.conv3(x).to('cuda')
        batch_norm = nn.BatchNorm2d(x.shape[1]).double().to('cuda')
        x = self.pool(F.relu(batch_norm(x))).to('cuda')

        x = torch.flatten(x, 1).to('cuda') # flatten all dimensions except batch
        x = F.relu(self.fc1(x)).to('cuda')
        x = F.relu(self.fc2(x)).to('cuda')
        x = self.fc3(x).to('cuda')
        
        return x


# In[ ]:

class Multilabel_SpectrogramDataset(Dataset):
    def __init__(self, path, option, class_mapping, 
                 train=True, max_length=-1, regression = None):
        
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression

        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_multi_labels(self.index)
        self.feats = [read_spectrogram(os.path.join(p, f),option) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("float64")

    def get_files_multi_labels(self, txt):
        # Returns a list of file names and a list of their labels
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split(',') for l in fd.readlines()[1:]]
            
        files, labels = [], []
        for l in lines:
            label = (float(l[1]),float(l[2]),float(l[3]))
            labels.append(label)
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = int(l[0])
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        return len(self.labels)


# In[ ]:

multi_specs = Multilabel_SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/multitask_dataset",
        class_mapping = None , train = True , option = 'sp_only')


# In[ ]:

multi_train_loader, multi_val_loader = torch_train_val_split(multi_specs,
                                                    32 ,32, val_size=.33)


## Loss Function For Multitask Learning

# In[ ]:

class L1loss(nn.Module):
    def __init__(self, weights):
        super(L1loss, self).__init__()
        self.w = weights.to(device)
    
    def forward(self,inputs,targets):
        l1loss = torch.zeros(inputs.shape[0]).to(device)
        
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                a = abs(inputs[i][j] - targets[i][j])
                l1loss[i] += self.w[j]*a
          
        return torch.mean(l1loss)


# In[ ]:

def train_multi_cnn(model,train_loader,val_loader,filename,overfit_batch=False):
    patience = 10
    learning_rate = 1e-4  

    weights = torch.Tensor([1,1,1]).to(device)
    criterion = L1loss (weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_per_epoch = []
    val_loss_per_epoch = []
    
    model.train()
    model = model.double()
    
    if(overfit_batch == False):
        EPOCHS = 20
        for epoch in range(EPOCHS):
            val_loss = 0
            for (i,data) in enumerate(val_loader):
            
                X_batch, y_batch, len_batch = data # get the features and labels
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
                out = model(X_batch,len_batch).double() # forward pass
                
                loss = criterion(out, y_batch) # compute per batch loss 
                val_loss +=loss.detach().item()
        
            val_loss_per_epoch.append(val_loss/i)
            
            i = np.argmin(val_loss_per_epoch)
            if (i == epoch):
                best_model = model     
            if (epoch > i + patience):
                val_loss_per_epoch.pop(-1)
                print('Early stopping...')
                break
            
            running_average_loss = 0
            for (i,data) in enumerate(train_loader):
                X_batch, y_batch, len_batch = data # get the features and labels
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
            
                optimizer.zero_grad() # ALWAYS USE THIS!! 
                out = model(X_batch,len_batch).double() # forward pass
                    
                loss = criterion(out, y_batch) # compute per batch loss 
                loss.backward() # compure gradients based on the loss function
                optimizer.step() # update weights 
                l = loss.detach().item()
                running_average_loss += l
                if i % 10 == 0:
                    print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i,
                                        float(l)))
                
            running_average_loss = running_average_loss/(i+1)
            loss_per_epoch.append(running_average_loss)
    
    elif (overfit_batch == True):
        EPOCHS = 750
        X_batch , y_batch , len_batch = next(iter(train_loader))
        for epoch in range(EPOCHS):
            running_average_loss = 0
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            len_batch = len_batch.to(device)
            
            optimizer.zero_grad() # ALWAYS USE THIS!! 
            out = model(X_batch,len_batch).double() # forward pass
            loss = criterion(out, y_batch) # compute per batch loss 
            loss.backward() # compure gradients based on the loss function
            optimizer.step() # update weights 
            l = loss.detach().item()
            running_average_loss += l
            print("Epoch: {} \t Loss {}".format(epoch,
                                      float(l)))
                
            running_average_loss = running_average_loss
            loss_per_epoch.append(running_average_loss)
            
    pickle.dump(best_model, open(filename, 'wb'))  
    
    return loss_per_epoch,val_loss_per_epoch


# In[ ]:

cnn_mult = CNN_multitask().to('cuda')

cnn_mult_tr_loss, cnn_mult_val_loss = train_multi_cnn(cnn_mult,multi_train_loader, multi_val_loader,
                                                 filename = 'cnn_mult.sav',overfit_batch = False)


# In[ ]:

plot_loss(cnn_mult_tr_loss,cnn_mult_val_loss,overfit_batch = False)


# In[ ]:

def model_eval_multi_regression(val_loader,filename):
    model = pickle.load(open(filename, 'rb'))
    model.eval()
    pred_valence = []
    true_valence = []
    pred_energy = []
    true_energy =[]
    pred_dance = []
    true_dance = []
    
    total_preds = 0
    with torch.no_grad(): 
        for i, data in enumerate(val_loader):
            x_batch, y_batch,len_batch = data 
            total_preds += len(len_batch)
            x_batch = x_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            len_batch = len_batch.to('cuda')
            out = model(x_batch,len_batch).double().to('cuda')
            for j in range(len(out)):
                pred_valence.append(out[j][0])
                true_valence.append(y_batch[j][0])
                pred_energy.append(out[j][1])
                true_energy.append(y_batch[j][1])
                pred_dance.append(out[j][2])
                true_dance.append(y_batch[j][2])
        
    true_valence = torch.Tensor(true_valence)
    pred_valence = torch.Tensor(pred_valence)
    true_energy = torch.Tensor(true_energy)
    pred_energy = torch.Tensor(pred_energy)
    true_dance = torch.Tensor(true_dance)
    pred_dance = torch.Tensor(pred_dance)
    
    corr1, pval1 = stats.spearmanr(true_valence, pred_valence)
    corr2, pval2 = stats.spearmanr(true_energy, pred_energy)
    corr3, pval3 = stats.spearmanr(true_dance, pred_dance)
    
    return corr1,corr2,corr3


# In[ ]:

corr_v,corr_e,corr_d = model_eval_multi_regression(multi_val_loader,filename = 'cnn_mult.sav')

print('Spearman Correlation For Valence -> {}'.format(corr_v))
print('Spearman Correlation For Energy -> {}'.format(corr_e))
print('Spearman Correlation For Danceability -> {}'.format(corr_d))