import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import torch, torchvision
import math

class SVR:
    """ Support Vector regression with differential privacy

    Parameters
    ----------
    alg : string (default='objective_pertubation')
          Chooses the privacy algorithm. Either 'output_pertubation' for
           output noise or 'objective_pertubation' for loss function noise

    C : float (default=1.0)
          Softmargin parameter

    lambda_ : float (default=0.5)
          Regularization parameter

    epsilion : float (default=0.5)
          Privacy parameter

    huberconst : float (default=0.5)
          Constant value for huber loss      
    epsilon_insensitive : float (default=0.2)
          Insensitive value in loss function
    """
    def __init__(self, alg='objective_pertubation' , C= 1.0, lambda_ = 0.5, epsilon = 0.5, huberconst = 0.5, epsilon_insensitive = 0.2):
        self.alg = alg
        self.C = C
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.huberconst = huberconst
        self.weight = 0.0
        self.beta = None
        self.b = None
        self._support_vectors = None
        self.epsilon_insensitive = epsilon

    def noisevector(self, dim, rate_lambda):
        """Generates a noise vector following Laplace distribution.

        The distribution of norm is Erlang distribution with parameters (dim, 
        rate_lambda). For the direction, pick uniformly by sampling dim number of 
        i.i.d. Gaussians and normalizing them.

        Parameters
        ----------
        dim (int): Dimension of the noise vector.
            rate_lambda (float): epsion^(-rate_lambda*x) in Erlang distribution.
        
        Returns
        -------
        res (ndarray of shape (dim,)): Noise vector.
        
        References:
            https://ergodicity.net/2013/03/21/generating-vector-valued-noise-for-differential-privacy/
        """
        # generate norm, after Numpy version 1.17.0
        normn = np.random.default_rng().gamma(dim, 1 / rate_lambda, 1)
        # generate direction
        r1 = np.random.normal(0, 1, dim)
        n1 = np.linalg.norm(r1, 2)  # get the norm of r1
        r2 = r1 / n1  # normalize r1
        # get the result noise vector
        res = r2 * normn
        return res

    
    def __decision_function(self, X):
        print(X.shape)
        print(self.beta.shape)
        return X.dot(self.beta) + self.b
    
    def __margin(self, X, y):
        return np.abs(y - self.__decision_function(X))
    
    def  prepare_data(self, X):
        """"Adds one extra dimension for bias trick

        Parameters
        ----------
        x : (ndarray of shape (n_sample, n_feature))
            Features

        Returns
        -------
        data : (ndarray of shape (n_sample,n_features + 1))
              Features with bias trick 

        """
        data = np.c_[X, np.ones(X.shape[0])]
        return data    

    def softmargin(self, X, y):
        """ Softmargin SVR with output perturbation

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample, )):
            labels   
        """        
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
       #https://cs.adelaide.edu.au/~chhshen/teaching/ML_SVR.pdf
        lr = 1e-3
        for _ in range(500):
            print(_)
            margin = self.__margin(X, y)
            helper = X
            for i in range(len(X)):
                if margin[i] < 0.2:
                    helper[i] = 0
            misclassified_pts_idx = np.where(margin < 0.2)[0]
            #d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            d_beta = self.beta - self.C * helper
            #d_beta = d_beta + 0.001 * (np.linalg.norm(d_beta,2)**2)
            self.beta = self.beta - lr * d_beta
            d_b = - self.C * 1#np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

            #loss = self.__cost(margin)

        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]

    def softmargin_with_noise(self, X, y):
        """ Softmargin SVR with output perturbation

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample, )):
            labels   
        """        
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
       #https://cs.adelaide.edu.au/~chhshen/teaching/ML_SVR.pdf
        lr = 1e-3
        for _ in range(500):
            margin = self.__margin(X, y)
            misclassified_pts_idx = np.where(margin < 0)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            #d_beta = d_beta + 0.001 * (np.linalg.norm(d_beta,2)**2)
            self.beta = self.beta - lr * d_beta
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

            #loss = self.__cost(margin)

        scale = len(y) * self.lambda_ * self.epsilon / 2
        l = len( X[0] )
        self.beta = self.beta + self.noisevector(l, scale)
        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]
#https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2
    def softmargin_with_obj(self, X, y):
        """ Softmargin SVR with objective perturbation

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample, )):
            labels   
        """        
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
        l = len( X[0] )
        N = len( y )
        c = 1 / ( 2 * 0.5 )
        c = 0.0
        tmp = c / (N * self.lambda_)
        Epsilonp = self.epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
        Delta = 0.0
        if Epsilonp > 0:
            Delta = 0
        else:
                Delta = c / ( N * (np.exp(self.epsilon/4)-1) ) - self.lambda_
                Epsilonp = self.epsilon / 2
        noise = self.noisevector(l, Epsilonp/2)
        lr = 1e-3
        for _ in range(500):
            margin = self.__margin(X, y)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            d_beta = d_beta + self.lambda_ * (np.linalg.norm(d_beta,2)**2)
            d_beta = d_beta + (1/N) * np.dot(noise,d_beta) + 0.5*Delta*(np.linalg.norm(d_beta,2)**2)
            self.beta = self.beta - lr * d_beta
            #self.beta = self.beta + (1/N) * np.dot(noise,self.beta) + 0.5*Delta*(np.linalg.norm(self.beta,2)**2)
            #self.beta = self.beta + 0.1 * (np.linalg.norm(self.beta,2)**2)
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b
            
        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]
    
    def loss(self, X):
        """ SVR loss function

        Parameters
        ----------
        X 
        """
        if (X < 0.2):
            return 0
        else:
            return X - 0.2    

    #https://ttic.uchicago.edu/~shai/papers/DekelShSi03.pdf
    def smoothloss(self, X):
        """" Smooth loss function for SVR

        Parameters
        ----------
        X: float
            distance to hyperplane

        Returns
        -------
        loss: float
            calculated smooth epsilon insensitive loss
        """
        return np.log(1 + np.exp(X-self.epsilon_insensitive)) + np.log(1 + np.exp(-X-self.epsilon_insensitive)) - 2*np.log(1+np.exp(-self.epsilon_insensitive))

    def hardmargin_output(self, X, Y):
        """"Hardmargin SVR with output perturbation

        Parameters
        ----------
        X : (ndarray of shape (n_sample, n_feature))
            Features

        y : (ndarray of shape (n_sample,))
            Labels

        Returns
        -------
        fpriv : (ndarray of shape (n_features,))
            Weights and bias        
        """
        X = self.prepare_data(X)
        N = len( Y )
        l = len( X[0] )#length of a data point
        scale = N * self.lambda_ * self.epsilon / 2
        noise = self.noisevector(l, scale)
        x0 = np.zeros(l)#starting point with same length as any data point

        def obj_func(x):
            jfd = self.smoothloss( Y[0] - np.dot(X[0],x))
            for i in range(1,N):
                jfd = jfd + self.smoothloss( abs(Y[i] - np.dot(X[i],x)))
            f = ( 1/N )*jfd + (1/2) * self.lambda_ * ( np.linalg.norm(x,2)**2 )
            return f

        #minimization procedure
        f = minimize(obj_func, x0, method='BFGS').x #empirical risk minimization using scipy.optimize minimize function
        fpriv = f + noise
        return fpriv[:-1], fpriv[-1]
    
    def hardmargin_objective(self, X, Y):
        """"Hardmargin SVR with objective perturbation

        Parameters
        ----------
        X : (ndarray of shape (n_sample, n_feature))
            Features

        y : (ndarray of shape (n_sample,))
            Labels

        Returns
        -------
        fpriv : (ndarray of shape (n_features,))
            Weights and bias        
        """
        X = self.prepare_data(X)
        N = len( Y )
        l = len( X[0] )#length of a data point
        scale = N * self.lambda_ * self.epsilon / 2
        noise = self.noisevector(l, scale)
        x0 = np.zeros(l)#starting point with same length as any data point
        c = 0.0
        tmp = c / (N * self.lambda_)
        Epsilonp = self.epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
        if Epsilonp > 0:
            Delta = 0
        else:
            Delta = c / ( N * (np.exp(self.epsilon/4)-1) ) - self.lambda_
            Epsilonp = self.epsilon / 2
        noise = self.noisevector(l, Epsilonp/2)

        def obj_func(x):
            jfd = self.smoothloss( abs(Y[0] - np.dot(X[0],x)))
            for i in range(1,N):
                jfd = jfd + self.smoothloss( abs(Y[i] - np.dot(X[i],x)))
            f = (1/N) * jfd + (1/2) * self.lambda_ * (np.linalg.norm(x,2)**2) + (1/N) * np.dot(noise,x) + (1/2)*Delta*(np.linalg.norm(x,2)**2)
            return f

        #minimization procedure
        f = minimize(obj_func, x0, method='BFGS').x #empirical risk minimization using scipy.optimize minimize function
        return f[:-1], f[-1]
    
    def torch_decision_function(self, X):
        """ Decision function for softmargin pytorch SVM

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        Returns
        -------
        result (ndarray of shape (n_sample,)):
            Decision values    
        """
        return X.matmul(self.beta) + self.b
    
    def smooth_svrtorch(self, X, y):
        loss = torch.abs(y - X.matmul(self.beta + self.b))
        result = torch.log((1 + torch.exp(loss-self.epsilon_insensitive))) + torch.log((1 + torch.exp(-loss-self.epsilon_insensitive))) - 2*np.log(1+np.exp(-self.epsilon_insensitive))
        return result
    
    def softmargin_clipping(self, X, y):
        """ Softmargin SVM with huberloss

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample, )):
            labels   
        """      
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.beta = torch.tensor(self.beta, requires_grad = True)
        self.b = torch.tensor([0.0], requires_grad = True)

        lr = 1#e-3
        clippingbound = 1.0
        epochs = 1000
        delta = 0.0001
        for _ in range(epochs):
            batchsize = 32
            idx = np.random.randint(self.n, size=batchsize)
            X_torch = torch.tensor(X[idx,:])
            y_torch = torch.tensor(y[idx])
            huber = self.smooth_svrtorch(X_torch, y_torch)
            huber.sum().backward()
            #print(self.beta.grad.shape)
            with torch.no_grad():
                update = lr * self.beta.grad
                if torch.linalg.norm(update) > 1.0:
                    update /= torch.linalg.norm(update)
                sigma = (batchsize/self.n) * (math.sqrt(epochs*np.log(1/delta))) * (clippingbound/self.epsilon)
                tensor_mean = torch.mean(update)
                rv = multivariate_normal(mean=tensor_mean.item(), cov=1*sigma)
                random_elements = rv.rvs(size=self.d)
                self.beta -= (update + torch.tensor(random_elements))
                
                self.b -= lr * self.b.grad
                #self.b.grad = torch.zeros(self.b.grad.shape)
                self.beta.grad.zero_()
                self.b.grad.zero_()

        self.beta = self.beta.detach().numpy()
        self.b = self.b.detach().numpy()

    def fit(self, X, Y):
        """ Regression performed on X.

        Parameters
        ----------
        X : (ndarray of shape (n_sample, n_feature))
            Features

        Y : (ndarray of shape (n_sample,))
            Labels

        """
        if (self.alg == 'objective'):
            self.weight, self.b = self.hardmargin_objective(X, Y)
        elif (self.alg == 'output'):
            self.weight, self.b = self.hardmargin_output(X, Y)
        elif (self.alg == 'new'):
            self.weight = self.hardmargin(X,Y)    
        elif (self.alg == 'clipping'):
            self.softmargin_clipping(X,Y)    
        else:
            print('This algorithm is not defined')   
             
    def get_weight(self):
        return self.beta

    def get_bias(self):
        return self.b    