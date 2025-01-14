import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class AdalineGD:
    """
    parameters
    learning rate is eta
    n_iter
    randomstate
    w_
    b_
    losses_ :MSE

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
            error = y-input_epoch
            self.w_ += self.eta*(-2/X.shape[0])* X.T.dot(error)
            self.b_ += self.eta*(-2)*np.mean(error)
            self.errors_.append(np.mean(error**2))
        return self
    def predict(self,X):
        return np.where(np.dot(X, self.w_)+self.b_>=0.5,1,0)

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

adaline_fit = AdalineGD().input(X,y)
adaline_pred = adaline_fit.predict(X_test)
print(adaline_pred)



class AdalineStochasticGD:
    """
     we are going to add two more things with the stochastic gradient descent
     1.a shuffle of examples
     2. a partial fit that adjust the weights, without starting over again, when 
     new data comes in
    """
    def __init__(self,eta = 0.01,n_iter =50,shuffle =True, random_state = 3011):
        self.n_iter = n_iter
        self.random_state = random_state
        self.eta = eta
        self.shuffle = shuffle 
    def _shuffle(self,X,y):
        shuffled_index = self.rgen.permutation(len(y))
        return X[shuffled_index], y[shuffled_index]
    def input(self,X,y):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01,size = X.shape[1])#mean = 0, std = 0.1, number of collumns of X
        self.b_ = np.float_(0.) # initial value of the bias =0
        self.errors_=[]
        for _ in range(self.n_iter):
            if self.shuffle ==True:
                X_shuffled,y_shuffled = self._shuffle(X,y)
            losses_placehold = []
            for xi,target in zip(X_shuffled,y_shuffled):
                losses_placehold.append(self._update_weights(xi,target))
            mse = np.mean(losses_placehold)
            self.errors_.append(mse)
        return self
        
    def _update_weights(self,X,y):
        input_epoch = np.dot(X, self.w_)+self.b_
        error = y-input_epoch
        self.w_ += self.eta * X.dot(error)
        self.b_ += self.eta * error
        losses = error**2
        return losses
    
    def new_fit(self, X,y):
        if y.ravel().shape[0]>1: # if there is more than one training example added
            for xi,target in zip(X_shuffled,y_shuffled):
                self._update_weights(X_shuffled,y_shuffled)
        else:
            self._update_weights(X_shuffled,y_shuffled)
    def predict(self,X):
        return np.where(np.dot(X, self.w_)+self.b_>=0.5,1,0)


adalineSGD_fit = AdalineStochasticGD().input(X,y)
adalineSGD_pred = adalineSGD_fit.predict(X_test)
print(adalineSGD_pred)

## The plot decision region is taken from the book 
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

##Visualization of the SGD
ada_sgd = AdalineStochasticGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.input(X, y)
plot_decision_regions(X, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()

plt.show()

plt.plot(range(1, len(ada_sgd.errors_) + 1), ada_sgd.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')


plt.show()