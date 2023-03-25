import os
from glob import glob
import librosa as lb
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy as sp
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pomegranate import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

def parse_fsdd(directory):
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("/")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = lb.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = lb.core.load(f, sr=None)

        return wav

    wavs = [read_wav(f) for f in files]

    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, ids, y, speakers

def extract_features_fsdd(wavs, n_mfcc=6, Fs=8000):
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        lb.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...", position = 0, leave = True)
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames

def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split ...")

    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def scale(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    for i in range(len(X_train)):
        X_train[i] = scaler.transform(X_train[i])
    return X_train


def parser(directory, n_mfcc=6):
    Fs = 8000
    wavs, ids, y, speakers = parse_fsdd(directory)
    frames = extract_features_fsdd(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

def data_partition(X_train, y_train):
    return train_test_split(X_train, y_train, stratify=y_train, test_size=0.20)

def digit_model(digit, X_train, y_train):
    print("Creating and training a GMM-HMM model for number {} ...".format(digit))
    X = [] # data from a single digit (can be a numpy array)
    data = [] # your data: must be a Python list that contains: 2D lists with the sequences (so its dimension would be num_sequences x seq_length x feature_dimension)
              # But be careful, it is not a numpy array, it is a Python list (so each sequence can have different length)

    for i in range(len(y_train)):
        if y_train[i] == digit:
            data.append(X_train[i])

    X = np.array(np.concatenate(data), dtype= 'float64')

    n_states = 4 # the number of HMM states
    n_mixtures = 4 # the number of Gaussians
    gmm = True # whether to use GMM or plain Gaussian

    dists = [] # list of probability distributions for the HMM states
    for i in range(n_states):
        if gmm:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, X)
        else:
            a = MultivariateGaussianDistribution.from_samples(X)
        dists.append(a)

    trans_mat = [[0.5, 0.5, 0.0, 0.0],[0.0, 0.5, 0.5, 0.0],[0.0, 0.0, 0.5, 0.5],[0.0, 0.0, 0.0, 1.0]] # your transition matrix
    starts = [1.0, 0.0, 0.0, 0.0] # your starting probability matrix
    ends = [0.0, 0.0, 0.0, 1.0] # your ending probability matrix

    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

    model.fit(data, max_iterations = 200)

    return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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

X_train, X_test, y_train, y_test, spk_train, spk_test = parser("recordings", n_mfcc = 13)
X_train = scale(X_train)
X_test = scale(X_test)
training_data, validation_data, training_digits, validation_digits = data_partition(X_train, y_train)


model0 = digit_model(0, training_data, training_digits)
model1 = digit_model(1, training_data, training_digits)
model2 = digit_model(2, training_data, training_digits)
model3 = digit_model(3, training_data, training_digits)
model4 = digit_model(4, training_data, training_digits)
model5 = digit_model(5, training_data, training_digits)
model6 = digit_model(6, training_data, training_digits)
model7 = digit_model(7, training_data, training_digits)
model8 = digit_model(8, training_data, training_digits)
model9 = digit_model(9, training_data, training_digits)


y_pred = []
print("Evaluating the model with the validation data set ...")
for i in range(len(validation_data)):
    logps = []
    sample = validation_data[i]
    logp0, _ = model0.viterbi(sample) 
    logp1, _ = model1.viterbi(sample)
    logp2, _ = model2.viterbi(sample)
    logp3, _ = model3.viterbi(sample)
    logp4, _ = model4.viterbi(sample)
    logp5, _ = model5.viterbi(sample)
    logp6, _ = model6.viterbi(sample)
    logp7, _ = model7.viterbi(sample)
    logp8, _ = model8.viterbi(sample)
    logp9, _ = model9.viterbi(sample)
    logps.append(logp0)
    logps.append(logp1)
    logps.append(logp2)
    logps.append(logp3)
    logps.append(logp4)
    logps.append(logp5)
    logps.append(logp6)
    logps.append(logp7)
    logps.append(logp8)
    logps.append(logp9)
    y_pred.append(np.argmax(logps))


print("Accuracy = :", accuracy_score(validation_digits, y_pred)) 
cm = confusion_matrix(validation_digits, y_pred)
plt.figure(1)
plot_confusion_matrix(cm, np.arange(10)) 

y_pred = []
print("Evaluating the model with the test data set ...")
for i in range(len(X_test)):
    logps = []
    sample = X_test[i]
    logp0, _ = model0.viterbi(sample) 
    logp1, _ = model1.viterbi(sample)
    logp2, _ = model2.viterbi(sample)
    logp3, _ = model3.viterbi(sample)
    logp4, _ = model4.viterbi(sample)
    logp5, _ = model5.viterbi(sample)
    logp6, _ = model6.viterbi(sample)
    logp7, _ = model7.viterbi(sample)
    logp8, _ = model8.viterbi(sample)
    logp9, _ = model9.viterbi(sample)
    logps.append(logp0)
    logps.append(logp1)
    logps.append(logp2)
    logps.append(logp3)
    logps.append(logp4)
    logps.append(logp5)
    logps.append(logp6)
    logps.append(logp7)
    logps.append(logp8)
    logps.append(logp9)
    y_pred.append(np.argmax(logps))


print("Accuracy = :", accuracy_score(y_test, y_pred)) 

cm = confusion_matrix(y_test, y_pred)
plt.figure(2)
plot_confusion_matrix(cm, np.arange(10))
