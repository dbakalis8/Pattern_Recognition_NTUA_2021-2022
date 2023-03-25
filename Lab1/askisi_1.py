import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import random
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import KFold,cross_val_score,learning_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier,BaggingClassifier

def show_sample(X,index):
    
    ex_1 = X[index,:]
    ex_1_1 = np.reshape(ex_1,(16,16))
    plt.imshow(ex_1_1)
    
    return 

def show_digits_samples(X,y):
    a=[]
    ind = np.zeros(10)
    while(len(a) < 10):
        n=random.randint(100,y.size-1)
        num = y[n]
        if a.count(num)==0:
            a.append(num)
            ind[len(a)-1]= n
            
    fig=plt.figure()
    
    for i in range(5):
        for j in range(2):
            index = int(ind[2*i+j])
            k = X[index,:]
            k_1 = np.reshape(k,(16,16))
            fig.add_subplot(5,2,2*i+j+1)
            plt.imshow(k_1)
            
    plt.show()
    return 

def digit_mean_at_pixel(X, y, digit, pixel):
    summ=0
    counter = 0 
    val=16*list(pixel)[0]+list(pixel)[1]
    
    for i in range(y.size):
        if(y[i]==digit):
             summ = summ + X[i,val]
             counter = counter + 1
    mean = summ / counter
    
    return mean

def digit_variance_at_pixel(X, y, digit, pixel):
    
    mean = digit_mean_at_pixel(X,y,digit,pixel)
    val=16*list(pixel)[0]+list(pixel)[1]
    summ = 0
    counter = 0
    for i in range (y.size):
        if(y[i]==digit):
            counter = counter + 1
            a = np.power((X[i,val] - mean),2)
            summ = summ +a
            
    variance = summ / counter
    
    return variance

def digit_mean(X, y, digit):
    a = np.zeros(X.shape[1])
    cntr = 0
    for j in range(y.size):
        if(y[j]==digit):
            a = a + X[j,:]
            cntr = cntr + 1
    a = a/cntr
    
    return a
   
def digit_variance(X, y, digit):
    
    a = np.zeros(X.shape[1]) 
    for i in range(X.shape[1]):
        a1 = i // 16
        b1 = i - (a1*16)
        pixel = (a1,b1)
        a[i]=digit_variance_at_pixel(X,y,digit,pixel)

    return a
  
def euclidean_distance(s, m):
    
    res = np.sqrt(sum (np.power((s-m),2)))
    
    return res

def euclidean_distance_classifier(X, X_mean):
    
    a=[]
    n = np.shape(X_mean)[0]
    
    for i in range(X.shape[0]):
        dis = np.zeros(n)
        for j in range(n):
            dis[j] = euclidean_distance(X[i,:],X_mean[j,:])
        h = np.argmin(dis)
        a.append(h)
    return a   

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        classes = len(unique_labels(y))
        if(self.X_mean_ == None):
            self.X_mean_ = np.zeros((classes,np.shape(X)[1]))
        self.X_mean_ = np.zeros((classes,np.shape(X)[1]))    
        for i in range(classes):
            self.X_mean_[i,:] = digit_mean(X,y,i)
        return self

    def predict(self, X):
        self.predictions = euclidean_distance_classifier(X,self.X_mean_)
        return self.predictions

    def score(self, X, y):   
        pr = self.predict(X)
        self.cntr = 0
        for i in range(len(y)):
            if(pr[i] != y[i]):
                self.cntr += 1
                
        self.success_rate = ( 1 - ( self.cntr/len(y) ) ) 
    
        return self.success_rate
   
def plot_clf(clf, X, y, labels):
    
    fig, ax = plt.subplots()
    fig1, bx = plt.subplots()

    title = ('Decision surface of Classifier')
 
    clf.fit(X,y)
    
    X0, X1 = X[:,0],X[:,1]
    x_min, x_max = X0.min() - 1 , X0.max() + 1
    y_min, y_max = X1.min() - 1 , X1.max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    color = ['blue','red','green','orange','black','teal','brown','yellow','chocolate','white','darkblue','teal']
    out = ax.contourf(xx, yy, Z, colors=color, levels=12 , alpha=0.8)
    out1 = bx.contourf(xx, yy, Z, colors=color , levels =12 , alpha=0.8)
    
    
    zero = ax.scatter(
        X0[y == 0], X1[y == 0],
        c='blue', label="0",
        s=60, alpha=0.9, edgecolors='k')
    
    one = ax.scatter(
        X0[y == 1], X1[y == 1],
        c='red', label="1", 
        s=60, alpha=0.9, edgecolors='k')
    
    two = ax.scatter(
        X0[y == 2], X1[y == 2],
        c='green', label='2', 
        s=60, alpha=0.9, edgecolors='k')
    
    three = ax.scatter(
        X0[y == 3], X1[y == 3],
        c='orange', label='3', 
        s=60, alpha=0.9, edgecolors='k')
    
    four = ax.scatter(
        X0[y == 4], X1[y == 4],
        c='black', label='4', 
        s=60, alpha=0.9, edgecolors='k')
    
    five = ax.scatter(
        X0[y == 5], X1[y == 5],
        c='brown', label='5', 
        s=60, alpha=0.9, edgecolors='k')
    
    six = ax.scatter(
        X0[y == 6], X1[y == 6],
        c='yellow', label='6', 
        s=60, alpha=0.9, edgecolors='k')
    
    seven = ax.scatter(
        X0[y == 7], X1[y == 7],
        c='chocolate', label='7', 
        s=60, alpha=0.9, edgecolors='k')
    
    eight = ax.scatter(
        X0[y == 8], X1[y == 8],
        c='white', label='8', 
        s=60, alpha=0.9, edgecolors='k')
    
    nine = ax.scatter(
        X0[y == 9], X1[y == 9],
        c='teal', label='9', 
        s=60, alpha=0.9, edgecolors='k')
    
    ax.set_ylabel("Feature 1")
    ax.set_xlabel("Feature 2")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()
    
    return 

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    return 

def calculate_priors(X, y):
    res = np.zeros(len(unique_labels(y)))
    for i in range(y.size):
        a = int(y[i])
        res[a] = res[a]+1
    
    res = res / y.size
    return res    
  
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
        self.classes = len(unique_labels(y))
        self.priors = calculate_priors(X,y)
        
        self.X_mean_ = np.zeros((self.classes,np.shape(X)[1]))
        self.X_var_  = np.zeros((self.classes,np.shape(X)[1]))
        
        for  i in range(self.classes):
            self.X_mean_[i,:] = digit_mean(X, y, i)
            self.X_var_[i,:] = digit_variance(X, y, i)
            
        self.X_std_ = np.sqrt(self.X_var_)
        
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
            self.predictions.append(h)
        return self.predictions

    def score(self, X, y):
        
        pr = self.predict(X)
        self.cntr = 0
        for i in range(len(y)):
            if(pr[i] != y[i]):
                self.cntr += 1
        self.success_rate = ( 1 - ( self.cntr/len(y) ) ) 
    
        return self.success_rate
        
        
# Βήμα 1ο

mytest = np.loadtxt("test.txt")
mytrain = np.loadtxt("train.txt")

y_train = mytrain[:,0]
y_test = mytest[:,0]

x_train = mytrain[:,1:257]
x_test = mytest[:,1:257]


# Βήμα 2ο

show_sample(x_train,130)

# Βήμα 3ο

show_digits_samples(x_train, y_train)

# Βήμα 4ο

mean_0 =digit_mean_at_pixel(x_train, y_train, 0, (10,10))
print(mean_0)

# Βήμα 5ο

variance_0 = digit_variance_at_pixel(x_train, y_train, 0, (10,10))
print(variance_0)

# Βήμα 6ο - 7o

mean0 = digit_mean(x_train,y_train,0)
mean0 = np.reshape(mean0,(16,16))
plt.figure(3)
plt.imshow(mean0)
    
# Βήμα 8ο

var0 = digit_variance(x_train,y_train,0)
var0 = np.reshape(var0,(16,16))
plt.figure(4)
plt.imshow(var0)

# Βήμα 9ο

mean_values = np.zeros((10,256))
var_values = np.zeros((10,256))

for i in range (10):
    mean_values[i,:] = digit_mean(x_train,y_train,i)
    var_values[i,:] = digit_variance(x_train,y_train,i)  
 
fig1 = plt.figure(5)  
  
for i in range(5):
        for j in range(2):
            k = 2*i+j
            mean_val = mean_values[k,:]
            mean_val = np.reshape(mean_val,(16,16))
            fig1.add_subplot(5,2,2*i+j+1)
            plt.imshow(mean_val)

# Βήμα 10ο

sample_101 = x_test[101,:]
euc_dis_101 = np.zeros(10)

for i in range(10):
    euc_dis_101[i] = euclidean_distance(sample_101,mean_values[i,:])

res_101 = np.argmin(euc_dis_101)

if(res_101 == y_test[100]):
    print("The Classification was Correct")
    
else :
    print("The Classification was Incorrect")
     
# Βήμα 11ο

classification_results = euclidean_distance_classifier(x_test,mean_values)

cntr = 0
for i in range(y_test.size):
    if(y_test[i] != classification_results[i]):
        cntr = cntr + 1
        
success_rate = (1 - (cntr/y_test.size))*100

print("Success Rate is " , success_rate , "%" )

c = EuclideanDistanceClassifier()
c.fit(x_train,y_train)
new_score = c.score(x_test,y_test)
print(new_score)

# Bήμα 13ο
  
# Α
cross_score = cross_val_score(c, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(cross_score_value)



# Β
c1 = EuclideanDistanceClassifier()
pca = PCA(n_components=2)

x_new = pca.fit_transform(x_train)
labels = (0,1)

plot_clf(c1, x_new , y_train, labels)


# Γ

c2 = EuclideanDistanceClassifier()
c2.fit(x_train,y_train)
train_sizes, train_scores, test_scores = learning_curve(
    c2, x_train , y_train , cv=5, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1.0, 5))

plot_learning_curve(train_scores,test_scores, train_sizes, (0,1))


# Βήμα 14ο

prob = calculate_priors(x_train, y_train)

# Βήμα 15ο

c3 = CustomNBClassifier(False)
c3.fit(x_train,y_train)
score_nb = c3.score(x_test,y_test)
print(score_nb)

cross_score = cross_val_score(c3, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(cross_score_value)

c4 = GaussianNB()
c4.fit(x_train,y_train)
c4.predict(x_test)
score_nb_sk = c4.score(x_test,y_test)
print(score_nb_sk)

cross_score = cross_val_score(c4, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(cross_score_value)

# Βήμα 16ο

c5 = CustomNBClassifier(True)
c5.fit(x_train,y_train)
score_nb = c5.score(x_test,y_test)
print(score_nb)

cross_score = cross_val_score(c5, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(cross_score_value)

# Βήμα 17ο 

svc = SVC(kernel="rbf")
svc.fit(x_train,y_train)
svc.predict(x_test)
sc = svc.score(x_test,y_test)
cross_score = cross_val_score(svc, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(sc)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
neigh.predict(x_test)
sc1 = neigh.score(x_test,y_test)
cross_score = cross_val_score(neigh, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score) 
print(sc1)


# Βήμα 18ο

# Α
clf = CustomNBClassifier(True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))

clf1 = SVC(kernel = 'rbf',probability=True)
clf2 = SVC(kernel = 'linear',probability=True)
clf3 = KNeighborsClassifier(n_neighbors=1)
clf4 = EuclideanDistanceClassifier()
clf5 = CustomNBClassifier(True)
clf6 = GaussianNB()

eclf1 = VotingClassifier(estimators=[
         ('rb', clf1), ('cl', clf2), ('kn', clf3)], voting='soft',weights=[6,2,4])


eclf1 = eclf1.fit(x_train, y_train)

cross_score = cross_val_score(eclf1, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(cross_score_value)


# Β

clf = BaggingClassifier(base_estimator=EuclideanDistanceClassifier(),n_estimators=9).fit(x_train,y_train)
cross_score = cross_val_score(clf, x_train , y_train, cv=KFold(n_splits=5,shuffle=True,random_state=42), scoring="accuracy")
cross_score_value = np.mean(cross_score)
print(cross_score_value) 
clf1 = EuclideanDistanceClassifier()
clf1.fit(x_train,y_train)
clf1.predict(x_test)
clf.predict(x_test)
sc = clf.score(x_test,y_test)
sc1 = clf1.score(x_test,y_test)
