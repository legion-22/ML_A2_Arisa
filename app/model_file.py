from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from itertools import combinations_with_replacement

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)

    # user to choose between zeros initialization or xavier // init='zeros' or init='xavier'
    # add variable momentum
    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500, batch_size=50, cv=kfold, init='zeros', momentum=0.0):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.init       = init
        self.momentum   = momentum
        self.prev_step  = 0

    def mse(self, ytrue, ypred):
        ytrue = np.array(ytrue)  # แปลงเป็น numpy array ถ้ายังไม่เป็น
        if ytrue.ndim == 0:      # กรณี scalar ให้เปลี่ยนเป็น array หนึ่งมิติ
            ytrue = np.array([ytrue])
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    ### Add function r^2
    def r2(self, ytrue, ypred):
        ss_total = ((ytrue - ytrue.mean()) ** 2).sum()
        ss_residual = ((ytrue - ypred) ** 2).sum()
        return 1 - (ss_residual / ss_total)
    
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.inf

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            '''
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            '''

            X_cross_train = X_train.iloc[train_idx]
            y_cross_train = y_train.iloc[train_idx]
            X_cross_val   = X_train.iloc[val_idx]
            y_cross_val   = y_train.iloc[val_idx]

            
            ### Xavier Initialization or Zero
            if self.init == 'zeros':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.init == 'xavier':
                m = X_cross_train.shape[1]  #number of samples
                lower, upper = -1.0 / np.sqrt(m), 1.0 / np.sqrt(m)  #calculate the range for the weights
                self.theta = np.random.uniform(lower, upper, X_cross_train.shape[1])    #randomly pick weights within range
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    #X_cross_train = X_cross_train[perm]
                    #y_cross_train = y_cross_train[perm]

                    X_cross_train = X_cross_train.iloc[perm]
                    y_cross_train = y_cross_train.iloc[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train.iloc[batch_idx].values.reshape(1, -1)
                            y_method_train = y_cross_train.iloc[batch_idx]
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            #X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            #y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            X_method_train = X_cross_train.iloc[batch_idx].values.reshape(1, -1)
                            y_method_train = y_cross_train.iloc[batch_idx]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
            
    ### Add Momentum to Gradient Descent                
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)

        #self.theta = self.theta - self.lr * grad

        step = self.lr * grad
        self.theta = self.theta - step + self.momentum * self.prev_step #updated Momentum
        self.prev_step = step   #save the value of the last step

        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    ### Add Plot Feature Importance function
    def plot_feature_importance(self, feature_names=None):
        coef = self._coef()
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(coef))]  # if no name feature

        plt.figure(figsize=(10, 5))
        plt.barh(feature_names, coef, color='blue')
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method)

def generate_polynomial_features(X, degree=2):
    # Convert X to numpy array if it is a DataFrame
    if hasattr(X, "values"):
        X = X.values
        
    m, n = X.shape
    X_poly = [np.ones(m)]  # bias term

    for d in range(1, degree + 1):  # create features from x^1 to x^degree.
        for combo in combinations_with_replacement(range(n), d):
            new_feature = np.prod(X[:, combo], axis=1)
            X_poly.append(new_feature)

    return np.column_stack(X_poly)

class NoPenalty:
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return 0