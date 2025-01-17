import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LogisticRegression:
    """
    We only have to change the loss function and add the sigmoid activation function 

    """
    def __init__(self,eta = 0.01,n_iter =50, random_state = 3011):
        self.n_iter = n_iter
        self.random_state = random_state
        self.eta = eta
    def input(self,X,y):
        """
        we have to set a random weight and bias to sart out
        then update it based on MSE
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01,size = X.shape[1])#mean = 0, std = 0.1, number of collumns of X
        self.b_ = np.float_(0.) # initial value of the bias =0
        self.errors_=[]
        for _ in range(self.n_iter):
            input_epoch = np.dot(X, self.w_)+self.b_
            activate = self.activation(input_epoch)
            error = y-input_epoch
            self.w_ += self.eta*(-2/X.shape[0])* X.T.dot(error)
            self.b_ += self.eta*(-2)*np.mean(error)
            loss = (-y).dot(np.log(activate)-(1-y).dot(np.log(1-activate))) /X.shape[0]
            self.errors_.append(np.mean(error**2))
        return self
    def activation(self,z):
         return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    def predict(self,X):
        return np.where(np.dot(X, self.w_)+self.b_>=0.5,1,0)
    

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

############LOADING IN DATASET#############
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

#Selecting the categories and assign it as the target
y=df.iloc[0:100,4].values #.values convert it to a np array
y=np.where(y=='Iris-setosa',0,1)

#Extract the features
X= df.iloc[0:100,[0,2]].values

#xtest
X_test = np.array([
    [2.0, 2.0],  # Close to class 0
    [4.8, 3.0]   # Close to class 1
])


##Visualization of the SGD
X_train_01_subset = X[(y == 0) | (y == 1)]
y_train_01_subset = y[(y == 0) | (y== 1)]

lrgd = LogisticRegression(eta=0.3, n_iter=1000, random_state=1)
lrgd.input(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('figures/03_05.png', dpi=300)
plt.show()