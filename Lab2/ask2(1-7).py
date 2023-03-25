import os
from glob import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.fftpack import idct
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def parse_free_digits(directory):
    
    # Parse relevant dataset info
    
    files = glob(os.path.join(directory, "*.wav"))
    file_names = [f.split('\\')[1] for f in files]
    fnames = [f.split(".")[0] for f in file_names]
    num = [f.split("_")[0] for f in fnames]
    speakers = [f.split("_")[1] for f in fnames]
    
    
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, num, speakers


def extract_features(wavs, n_mfcc, Fs):
    
    # Extract MFCCs for all wavs
    
    window = 25 * Fs // 1000 
    step = 10 * Fs // 1000
    frames = [librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames

def melspectograms(wavs, n_mels, Fs):
    
    # Extract MFSCs for all wavs
    
    window = 25 * Fs // 1000 
    step = 10 * Fs // 1000
    mfsc = [librosa.feature.melspectrogram(
        wav, Fs, n_fft=window, hop_length=window - step, n_mels=n_mels
        ).T

        for wav in tqdm(wavs, desc="Extracting mfsc features...")
    ]
    
    return mfsc

def zerocrossrate(wavs):
    zrc=[]
    
    for wav in wavs:
        k=librosa.feature.zero_crossing_rate(wav)
        zrc.append(k)    
    return zrc

def ccstft(wavs):
    ccstft=[]
    
    for wav in wavs:
        c=librosa.feature.chroma_stft(y=wav, sr=16000)
        ccstft.append(c)
    
    return ccstft

def mean_std_feature(mfcc , mfcc_delta , mfcc_delta2):
    mean_val = []
    std_val = []
   
    for i in range(len(mfcc)):
        v = np.concatenate((mfcc[i],mfcc_delta[i],mfcc_delta2[i]))
        mean = np.zeros(np.shape(v)[1])
        std = np.zeros(np.shape(v)[1])
        
        for j in range(np.shape(v)[1]):
            mean[j] = np.mean(v[:,j])
            std[j] = np.std(v[:,j])
        
        mean_val.append(mean)
        std_val.append(std)
         
    return mean_val,std_val


def plot_scatter_2d(arr,number,speaker,fig,string):
    
    sp = ['.','^','o','v','s','p','*','h','+','x','D','1','2','P','8',',']
    num = ['black','blue','crimson','darkgreen','orange','red','yellow','purple','brown']
    
    plt.figure(fig)
    for i in range(len(speaker)):         
        arr_sp = arr[i]
        sp_int = int(speaker[i])
        
        if(number[i]=='one'):
            n=0
        elif(number[i]=='two'):
            n=1
        elif(number[i]=='three'):
            n=2
        elif(number[i]=='four'):
            n=3
        elif(number[i]=='five'):
            n=4
        elif(number[i]=='six'):
            n=5
        elif(number[i]=='seven'):
            n=6
        elif(number[i]=='eight'):
            n=7
        elif(number[i]=='nine'):
            n=8
         
        plt.scatter(arr_sp[0],arr_sp[1],marker=sp[n],c=num[n])
        
    if(string == 'Mean'):
        plt.title('Scatter plot for 2 dimensions of Mean values')   
        plt.xlabel('Mean Value 1')
        plt.ylabel('Mean Value 2')
        plt.show()
        
    elif(string == 'Std'):
        plt.title('Scatter plot for 2 dimensions of Std values')   
        plt.xlabel('Std Value 1')
        plt.ylabel('Std Value 2')
        plt.show()
    
    elif(string == 'PCA Mean'):
        plt.title('Scatter plot for 2 dimensions of Mean values (PCA)')   
        plt.xlabel('Mean Value 1')
        plt.ylabel('Mean Value 2')
        plt.show()
        
    elif(string == 'PCA Std'):
        plt.title('Scatter plot for 2 dimensions of Std values (PCA)')   
        plt.xlabel('Std Value 1')
        plt.ylabel('Std Value 2')
        plt.show()
    return


def plot_scatter_3d(arr,number,speaker,fig,string):

    sp = ['.','^','o','v','s','p','*','h','+','x','D','1','2','P','8',',']
    num = ['black','blue','crimson','darkgreen','orange','red','yellow','purple','brown']
    fig = plt.figure(fig)
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(speaker)):         
        arr_sp = arr[i]
        sp_int = int(speaker[i])
        
        if(number[i]=='one'):
            n=0
        elif(number[i]=='two'):
            n=1
        elif(number[i]=='three'):
            n=2
        elif(number[i]=='four'):
            n=3
        elif(number[i]=='five'):
            n=4
        elif(number[i]=='six'):
            n=5
        elif(number[i]=='seven'):
            n=6
        elif(number[i]=='eight'):
            n=7
        elif(number[i]=='nine'):
            n=8
         
        ax.scatter(arr_sp[0],arr_sp[1],arr_sp[2],marker=sp[n],c=num[n])
        
    if(string == 'Mean'):
        plt.title('Scatter plot for 2 dimensions of Mean values')   
        plt.xlabel('Mean Value 1')
        plt.ylabel('Mean Value 2')
        plt.show()
        
    elif(string == 'Std'):
        plt.title('Scatter plot for 2 dimensions of Std values')   
        plt.xlabel('Std Value 1')
        plt.ylabel('Std Value 2')
        plt.show()
    
    elif(string == 'PCA Mean'):
        plt.title('Scatter plot for 2 dimensions of Mean values (PCA)')   
        plt.xlabel('Mean Value 1')
        plt.ylabel('Mean Value 2')
        plt.show()
        
        
    elif(string == 'PCA Std'):
        plt.title('Scatter plot for 2 dimensions of Std values (PCA)')   
        plt.xlabel('Std Value 1')
        plt.ylabel('Std Value 2')
        plt.show()
    return


def split_free_digits(wavs,speakers, labels,percentage):
    print("Splitting in train test split using the default dataset split")
    lenght = int(percentage*len(wavs))
    X_train, y_train, spk_train = [], [], []
    index = []
    X_test, y_test, spk_test = [], [], []

    while(len(X_train)<lenght):
        n = random.randint(1,len(wavs)-1)
        if(index.count(n)==0):
            index.append(n)
            X_train.append(wavs[n])
            y_train.append(labels[n])
            spk_train.append(speaker[n])
            
    for i in range(len(wavs)):
        if(index.count(i)==0):
            X_test.append(wavs[i])
            y_test.append(labels[i])
            spk_test.append(speaker[i])
            
    return X_train, X_test, y_train, y_test, spk_train, spk_test

def calculate_priors(y):
    labels , counts = np.unique(y_train,return_counts=True)
    counts = counts / len(y)
    return counts    
  
def normal_dist(x , mean , sd):
    for i in range(sd.size):
        if(sd[i]==0):
           # sd[i] = 1e-4 * max(sd)
           sd[i] = 1e-4
    prob_density = (1/sd*(np.sqrt(2*np.pi))) * np.exp(-0.5*(((x-mean)/sd)**2))

    return prob_density
  
class CustomNBClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, use_unit_variance):
        self.use_unit_variance = use_unit_variance


    def fit(self, X, y):
        self.cl = unique_labels(y)
        self.classes = len(unique_labels(y))
        self.priors = calculate_priors(y)
        
        self.X_mean_ = np.zeros((self.classes,np.shape(X)[1]))
        self.X_std_  = np.zeros((self.classes,np.shape(X)[1]))
        
        for  i in range(self.classes):
            val=[]
            for j in range(len(y)):
                if(y[j]==self.cl[i]):
                    val.append(X[j])
            m=[]
            s=[]
            val = np.array(val)
            for j in range(np.shape(X)[1]):
                m.append(np.mean(val[:,j]))
                s.append(np.std(val[:,j]))
                
            self.X_mean_[i,:] = m
            self.X_std_[i,:] = s
            
        if(self.use_unit_variance == True):
            self.X_std_ = np.ones((self.classes,np.shape(X)[1]))
        
        return self

    def predict(self, X):
        
        self.predictions = []
        for i in range(np.shape(X)[0]):
            lik = np.zeros(self.classes)
            for j in range(len(lik)):
                s1 = normal_dist(X[i,:], self.X_mean_[j,:], self.X_std_[j,:])
                lik[j] = np.prod(s1)*self.priors[j]
            h = np.argmax(lik)
            self.predictions.append(self.cl[h])
        return self.predictions

    def score(self, X, y):
        
        pr = self.predict(X)
        self.cntr = 0
        for i in range(len(y)):
            if(pr[i] != y[i]):
                self.cntr += 1
        self.success_rate = ( 1 - ( self.cntr/len(y) ) ) 
        self.success_rate = np.round(self.success_rate,3)
        return self.success_rate
    
######### MAIN ###############

# Βήμα 2ο : Δημιουργία Λίστας με αρχεία ήχου , ομιλητή (speakers) και ψηφίο (number)

wavs , Fs , number , speaker = parse_free_digits("digits")

# Βήμα 3ο : Εξαγωγή 13 χαρακτηριστικών ανά αρχείο ήχου

mfcc = extract_features(wavs , 13 , Fs )

mfsc = melspectograms(wavs, 13 , Fs)

# 1η Τοπική Παράγωγος

mfcc_delta = [librosa.feature.delta(mf) for mf in mfcc]

# 2η Τοπική Παράγωγος

mfcc_delta2 = [librosa.feature.delta(mf,order=2) for mf in mfcc]

# Βήμα 4ο : 

# Ιστόγραμματα 1ων και 2ων MFCC για τα ψηφία 3 και 4

mfcc_4 = []
mfcc_3 = []
mfsc_3 = []
mfsc_4 = []

for i in range (len(mfcc)):
    if(number[i] == 'four'):
        mfcc_4.append(mfcc[i])
        mfsc_4.append(mfsc[i])
        
    elif(number[i] == 'three'):
        mfcc_3.append(mfcc[i])
        mfsc_3.append(mfsc[i])        

g3 = np.concatenate(mfcc_3)
g4 = np.concatenate(mfcc_4)

s31 = g3[:,0]
s32 = g3[:,1]
s41 = g4[:,0]
s42 = g4[:,1]


plt.figure(0)
plt.hist(s31,bins='auto',color='#0504aa',alpha=0.5, rwidth=0.85)
plt.grid(True)
plt.xlabel("Frequency")
plt.ylabel("MFCC Values")
plt.title('Histogram of 1st MFCC for number 3')

plt.figure(1)
plt.hist(s32,bins='auto',color='#0504aa',alpha=0.5, rwidth=0.85)
plt.grid(True)
plt.xlabel("Frequency")
plt.ylabel("MFCC Values")
plt.title('Histogram of 2st MFCC for number 3')

plt.figure(2)
plt.hist(s41,bins='auto',color='#0504aa',alpha=0.5, rwidth=0.85)
plt.grid(True)
plt.xlabel("Frequency")
plt.ylabel("MFCC Values")
plt.title('Histogram of 1st MFCC for number 4')

plt.figure(3)
plt.hist(s42,bins='auto',color='#0504aa',alpha=0.5, rwidth=0.85)
plt.grid(True)
plt.xlabel("Frequency")
plt.ylabel("MFCC Values")
plt.title('Histogram of 2st MFCC for number 4')

# Εξαγωγή των MFSC 2 εκφωνήσεων απο 2 ομιλητές για τα ψηφία 3 και 4

th_1 = mfcc_3[0]
th_2 = mfcc_3[6]

fo_1 = mfcc_4[0]
fo_2 = mfcc_4[6]

R31 = np.corrcoef(th_1.T)
R32 = np.corrcoef(th_2.T)
R41 = np.corrcoef(fo_1.T)
R42 = np.corrcoef(fo_2.T)

th_1 = mfsc_3[0]
th_2 = mfsc_3[6]

fo_1 = mfsc_4[0]
fo_2 = mfsc_4[6]

P31 = np.corrcoef(th_1.T)
P32 = np.corrcoef(th_2.T)
P41 = np.corrcoef(fo_1.T)
P42 = np.corrcoef(fo_2.T)


plt.matshow(R31)
plt.title("Covariance of MFCC for Number 3 from Speaker 1")

plt.matshow(R32)
plt.title("Covariance of MFCC for Number 3 from Speaker 7")

plt.matshow(R41)
plt.title("Covariance of MFCC for Number 4 from Speaker 1")

plt.matshow(R42)
plt.title("Covariance of MFCC for Number 4 from Speaker 7")

plt.matshow(P31)
plt.title("Covariance of MFSC for Number 3 from Speaker 1")

plt.matshow(P32)
plt.title("Covariance of MFSC for Number 3 from Speaker 7")

plt.matshow(P41)
plt.title("Covariance of MFSC for Number 4 from Speaker 1")

plt.matshow(P42)
plt.title("Covariance of MFSC for Number 4 from Speaker 7")


# Βήμα 5ο: Εξαγωγή Μοναδικού Διανύσματος Χαρακτηριστικών για κάθε εκφώνηση

mean , std = mean_std_feature(mfcc,mfcc_delta,mfcc_delta2)

# Scatter plot των 2 πρώτων διαστάσεων των διανυσμάτων αυτών

plot_scatter_2d(mean,number,speaker,12,'Mean')
plot_scatter_2d(std,number,speaker,13,'Std')

# Βήμα 6ο: PCA

pca1 = PCA(n_components=2)
pca2 = PCA(n_components=2)
pca3 = PCA(n_components=3)
pca4 = PCA(n_components=3)

mean_new = pca1.fit_transform(mean)
std_new = pca2.fit_transform(std)
mean_new1 = pca3.fit_transform(mean)
std_new1 = pca4.fit_transform(std)

plot_scatter_2d(mean_new, number, speaker, 14,'PCA Mean')
plot_scatter_2d(std_new, number, speaker, 15,'PCA Std')
plot_scatter_3d(mean_new1, number, speaker, 16, 'PCA Mean')
plot_scatter_3d(std_new1, number, speaker,17,'PCA Std')

print(pca1.explained_variance_ratio_)
print(pca2.explained_variance_ratio_)
print(pca3.explained_variance_ratio_)
print(pca4.explained_variance_ratio_)

# Βήμα 7ο: Classification

# Spliting The Data to Train and Test

x_train , x_test , y_train , y_test , spk_train , spk_test = split_free_digits(wavs, 
                            speaker, number, 0.7)
 
mfcc_train = extract_features(x_train,13, Fs)
mfcc_test = extract_features(x_test, 13, Fs)

mfcc_train_delta = [librosa.feature.delta(mf) for mf in mfcc_train]
mfcc_train_delta2 = [librosa.feature.delta(mf,order=2) for mf in mfcc_train]

mfcc_test_delta = [librosa.feature.delta(mf) for mf in mfcc_test]
mfcc_test_delta2 = [librosa.feature.delta(mf,order=2) for mf in mfcc_test]

mean_train , std_train = mean_std_feature(mfcc_train,mfcc_train_delta,mfcc_train_delta2)
mean_test, std_test = mean_std_feature(mfcc_test,mfcc_test_delta,mfcc_test_delta2)

# Standarizing Our Training Data

mean_train = np.array(mean_train)
std_train = np.array(std_train)

mean_test = np.array(mean_test)
std_test = np.array(std_test)

train_scaler_mean = StandardScaler()
train_scaler_std = StandardScaler()

test_scaler_mean = StandardScaler()
test_scaler_std = StandardScaler()

train_scaler_mean.fit(mean_train)
train_scaler_std.fit(std_train)

test_scaler_mean.fit(mean_test)
test_scaler_std.fit(std_test)

mean_train_norm = train_scaler_mean.transform(mean_train)
std_train_norm = train_scaler_std.transform(std_train)

mean_test_norm = test_scaler_mean.transform(mean_test)
std_test_norm = test_scaler_std.transform(std_test)

# Classification with Custom Naive Bayes (από 1η εργαστηριακή)

cnb = CustomNBClassifier(False)
cnb.fit(mean_train_norm,y_train)
score_cnb = cnb.score(mean_test_norm,y_test)
print("Custom Naive Bayes prediction score is: ",score_cnb)

# Classification with Sklearn Naive Bayes

nb = GaussianNB()
nb.fit(mean_train_norm,y_train)
nb.predict(mean_test_norm)
score_nb = nb.score(mean_test_norm,y_test)
print("Naive Bayes prediction score is: ",score_nb)

# Classification with Nearest Neighbor

nn = KNeighborsClassifier(n_neighbors=1)
nn.fit(mean_train_norm,y_train)
nn.predict(mean_test_norm)
score_nn = nn.score(mean_test_norm,y_test)
print("Nearest Neighbor prediction score is: ",score_nn)

# Classification with Logistic Regression

lr = LogisticRegression(random_state=0)
lr.fit(mean_train_norm,y_train)
lr.predict(mean_test_norm)
score_lr = lr.score(mean_test_norm,y_test)
print("Linear Regression prediction score is: ",score_lr)

# Classification with SVM (RBF)

svm = SVC(kernel="rbf")
svm.fit(mean_train_norm,y_train)
svm.predict(mean_test_norm)
score_svm = svm.score(mean_test_norm,y_test)
print("SVM (rbf) prediction score is: ",score_svm)

# Χρήση Zero Cross Rate για την βελτίωση των ταξινομητών

zrc_train = zerocrossrate(x_train)
ccstft_train = ccstft(x_train)
zrc_test = zerocrossrate(x_test)
ccstft_test = ccstft(x_test)

zrc_mean_train=[]
ccstft_mean_train=[]
zrc_mean_test=[]
ccstft_mean_test=[]

for i in range(len(zrc_train)):
    zrc_mean_train.append(np.mean(zrc_train[i]))
    ccstft_mean_train.append(np.mean(ccstft_train[i]))
    
for i in range(len(zrc_test)):    
    zrc_mean_test.append(np.mean(zrc_test[i]))
    ccstft_mean_test.append(np.mean(ccstft_test[i]))

zipped_mean_train = np.c_[mean_train_norm,zrc_mean_train,ccstft_mean_train]
zipped_mean_test = np.c_[mean_test_norm,zrc_mean_test,ccstft_mean_test]

cnb = CustomNBClassifier(False)
cnb.fit(zipped_mean_train,y_train)
score_cnb = cnb.score(zipped_mean_test,y_test)
print("Custom Naive Bayes prediction score is: ",score_cnb)

# Classification with Sklearn Naive Bayes

nb = GaussianNB()
nb.fit(zipped_mean_train,y_train)
nb.predict(zipped_mean_test)
score_nb = nb.score(zipped_mean_test,y_test)
print("Naive Bayes prediction score is: ",score_nb)

# Classification with Nearest Neighbor

nn = KNeighborsClassifier(n_neighbors=1)
nn.fit(zipped_mean_train,y_train)
nn.predict(zipped_mean_test)
score_nn = nn.score(zipped_mean_test,y_test)
print("Nearest Neighbor prediction score is: ",score_nn)

# Classification with Logistic Regression

lr = LogisticRegression(random_state=0)
lr.fit(zipped_mean_train,y_train)
lr.predict(zipped_mean_test)
score_lr = lr.score(zipped_mean_test,y_test)
print("Linear Regression prediction score is: ",score_lr)

# Classification with SVM (RBF)

svm = SVC(kernel="rbf")
svm.fit(zipped_mean_train,y_train)
svm.predict(zipped_mean_test)
score_svm = svm.score(zipped_mean_test,y_test)
print("SVM (rbf) prediction score is: ",score_svm)