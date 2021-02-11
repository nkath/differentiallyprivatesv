import numpy as np
import math
import copy
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
import torch, torchvision
import time

class BinarySVM:
    """ Binary SVM with privacy options

    Parameters
    ----------
    alg : string (default='objective_pertubation')
          Chooses the privacy algorithm. Either 'non_private' for no privacy, 'output_pertubation' for
           privacy or 'objective_pertubation' for privacy

    C : float (default=1.0)
          Regularization parameter. Must be greater than 0.0    

    lambda_ : float (default=0.5)
          Regularization parameter

    epsilion : float (default=0.5)
          Privacy parameter

    huberconst : float (default=0.5)
          Constant value for huber loss      
    """
    def __init__(self, alg='objective_pertubation', C= 1.0, lambda_ = 0.5, epsilon = 0.5, huberconst = 0.5):
        self.alg, self.C, self.lambda_ = alg, C, lambda_
        self.epsilon, self.huberconst = epsilon, huberconst
        self.weights = 0
        self.bias = 0.0
        self.beta = None
        self.b = None
        self._support_vectors = None


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

    def noisevector(self, dim, rate_lambda):
        """Generates a noise vector following Laplace distribution.

        The distribution of norm is Erlang distribution with parameters (dim, 
        rate_lambda). For the direction, pick uniformly by sampling dim number of 
        i.i.d. Gaussians and normalizing them.

        Args:
            dim (int): Dimension of the noise vector.
            rate_lambda (float): epsion^(-rate_lambda*x) in Erlang distribution.
        
        Returns:
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

    def huberloss(self, z, huberconst):
        """Returns normal Huber loss (float) for a sample. 

        Args:
            z (float): x_i (1, n_feature) * y_i (int: -/+1) * w (n_feature, 1).
            huberconst (float): Huber loss parameter.        

        References:
            chaudhuri2011differentially: equation 7 & corollary 13
        """
        if z > 1.0 + huberconst:
            hloss = 0
        elif z < 1.0 - huberconst:
            hloss = 1 - z
        else:
            hloss = (1 + huberconst - z) ** 2 / (4 * huberconst)
        return hloss


    def eval_svm(self, weights, XY, num, lambda_, b, huberconst):
        """Evaluates differentially private regularized svm loss for a data set. 

        Args:
            weights (ndarray of shape (n_feature,)): Weights in w'x in classifier.
            XY (ndarray of shape (n_sample, n_feature)): 
                each row x_i (1, n_feature) * label y_i (int: -/+1).
            num (int): n_sample.
            lambda_ (float): Regularization parameter. 
            b (ndarray of shape (n_feature,)): Noise vector. If b is a zero vector,
                returns non-private regularized svm loss.
            huberconst (float): Huber loss parameter.
        
        Returns:
            fw (float): Differentially private regularized svm loss.
        """
        # add Huber loss from all samples
        XYW = np.matmul(XY, weights)
        fw = np.mean(
            [self.huberloss(z=z, huberconst=huberconst) for z in XYW], dtype=np.float64
        )
        # add regularization term (1/2 * lambda * |w|^2) and b term (1/n * b'w)
        fw += 0.5 * lambda_ * weights.dot(weights) + 1.0 / num * b.T.dot(weights)
        return fw


    def train_svm_nonpriv(self, XY, num, dim, lambda_, huberconst):
        """Trains a non-private regularized svm classifier. 

        Args:
            XY (ndarray of shape (n_sample, n_feature)): 
                each row x_i (1, n_feature) * label y_i (int: -/+1).
            num (int): n_sample.
            dim (int): Dimension of x_i, i.e., n_feature.
            lambda_ (float): Regularization parameter. 
            huberconst (float): Huber loss parameter.
        
        Returns:
            w_nonpriv (ndarray of shape (n_feature,)): 
                Weights in w'x in svm classifier.

        Raises:
            Exception: If the optimizer exits unsuccessfully.
        """
        w0 = np.zeros(dim)  # w starting point
        b = np.zeros(dim)  # zero noise vector
        res = minimize(
            self.eval_svm,
            w0,
            args=(XY, num, lambda_, b, huberconst),
            method="L-BFGS-B",
            bounds=None,
            tol=0.001
        )
        if not res.success:
            raise Exception(res.message)
        w_nonpriv = res.x
        return w_nonpriv


    def train_svm_outputperturb(self, XY, num, dim, lambda_, epsilon, huberconst):
        """Trains a private regularized svm classifier by output perturbation. 

        First, train a non-private svm classifier to get weights w_nonpriv, 
        then add noise to w_nonpriv to get w_priv.

        Args:
            XY (ndarray of shape (n_sample, n_feature)): 
                each row x_i (1, n_feature) * label y_i (int: -/+1).
            num (int): n_sample.
            dim (int): Dimension of x_i, i.e., n_feature.
            lambda_ (float): Regularization parameter. 
            epsilon (float): Privacy parameter.
            huberconst (float): Huber loss parameter.
        
        Returns:
            w_priv (ndarray of shape (n_feature,)): 
                Weights in w'x in svm classifier.
        
        References:
            chaudhuri2011differentially: Algorithm 1 output perturbation.
        """
        w_nonpriv = self.train_svm_nonpriv(
            XY=XY, num=num, dim=dim, lambda_=lambda_, huberconst=huberconst
        )
        beta = num * lambda_ * epsilon / 2
        noise = self.noisevector(dim, beta)
        w_priv = w_nonpriv + noise
        return w_priv


    def train_svm_objectiveperturb(self, XY, num, dim, lambda_, epsilon, huberconst):
        """Trains a private regularized svm classifier by objective perturbation. 

        Add noise to the objective (loss function).

        Args:
            XY (ndarray of shape (n_sample, n_feature)): 
                each row x_i (1, n_feature) * label y_i (int: -/+1).
            num (int): n_sample.
            dim (int): Dimension of x_i, i.e., n_feature.
            lambda_ (float): Regularization parameter. 
            epsilon (float): Privacy parameter.
            huberconst (float): Huber loss parameter.
        
        Returns:
            w_priv (ndarray of shape (n_feature,)): 
                Weights in w'x in svm classifier.

        Raises:
            Exception: If epsilon_p < 1e-4, where epsilon_p is calculated from 
                n_sample, lambda_ and epsilon.
            Exception: If the optimizer exits unsuccessfully.
        
        References:
            chaudhuri2011differentially: Algorithm 2 objective perturbation.
            http://cseweb.ucsd.edu/~kamalika/code/dperm/documentation.pdf
        """
        c = 1 / (2 * huberconst)  # value for svm
        tmp = c / (num * lambda_)
        epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
        if epsilon_p > 0.0:
            lambda_ = 0.0
        else:
            lambda_ = c / (num * (math.exp((epsilon_p / 4)) - 1)) - lambda_
            epsilon_p /= 2    

        w0 = np.zeros(dim)
        beta = epsilon_p / 2
        b = self.noisevector(dim, beta)
        res = minimize(
            self.eval_svm,
            w0,
            args=(XY, num, lambda_, b, huberconst),
            method="L-BFGS-B",
            bounds=None,
        )
        if not res.success:
            raise Exception(res.message)
        w_priv = res.x
        return w_priv

    def decision_svmhuber(self, intercept, X):
        """Private Function. Returns decision values for feature matrix X by SVM with huber loss.

        Args:
            intercept: float, intercept in the model y = w'x + intercept, 
                for unpreprocessed raw data.
            X (ndarray of shape (n_sample, n_feature)): Features.

        Returns:
            float ndarray of shape (n_sample,): w'x + intercept. 
            Used as y_score in sklearn.metrics.roc_auc_score.
        """
        return np.dot(X, self.weights) + self.bias    

    
    def huber(self, z, h):#chaudhuri2011differentially corollary 21
        """Returns normal Huber loss (float) for a sample. 

        Args:
            z (float): x_i (1, n_feature) * y_i (int: -/+1) * w (n_feature, 1).
            h (float): Huber loss parameter.        

        References:
            chaudhuri2011differentially: equation 7 & corollary 13
        """
        if z > 1 + h:
            hb = 0
        elif np.fabs(1-z) <= h:
            hb = (1 + h - z)**2 / (4 * h)
        else:
            hb = 1 - z
        return hb

    def gaussNoise(self, scale, size):
        return np.random.default_rng().normal(scale= scale, size = size)

    def svm_output_train(self, data, labels, epsilon, Lambda, h):
        data = self.prepare_data(data)
        N = len( labels )
        l = len( data[0] )#length of a data point
        scale = (2 * self.C * math.sqrt(2* np.log(1.25/0.0001)) * math.sqrt(21)) / self.epsilon
        #noise = self.noisevector(l, scale)
        noise = self.gaussNoise(scale, l)
        x0 = np.zeros(l)#starting point with same length as any data point
        def obj_func(x):
            jfd =  self.huber( labels[0] * np.dot(data[0],x), h )
            for i in range(1,N):
                jfd = jfd +  self.huber( labels[i] * np.dot(data[i],x), h )
            f = self.C * jfd + (1/2) * Lambda * ( np.linalg.norm(x,2)**2 )
            return f

        #minimization procedure
        f = minimize(obj_func, x0, method='BFGS').x #empirical risk minimization using scipy.optimize minimize function
        #print(noise)
        fpriv = f + noise
        return fpriv[:-1], fpriv[-1]

    def svm_objective_train(self, data, labels,  epsilon, Lambda, h):
        data = self.prepare_data(data)
        #parameters in objective perturbation method
        c = 1 / ( 2 * h )#chaudhuri2011differentially corollary 13
        N = len( labels )#number of data points in the data set
        l = len( data[0] )#length of a data point
        x0 = np.zeros(l)#starting point with same length as any data point
        tmp = c / (N * Lambda)
        scaled_epsilon = epsilon * (1/10)
        Epsilonp = scaled_epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
        Delta = 0.0
        if Epsilonp > 0:
            Delta = 0
        else:
            Delta = c / ( N * (np.exp(epsilon/4)-1) ) - Lambda
            Epsilonp = scaled_epsilon / 2
        noise = self.noisevector(l, Epsilonp/2)
        def obj_func(x):
            jfd = self.huber( labels[0] * np.dot(data[0], x), h)
            for i in range(1,N):
                jfd = jfd + self.huber( labels[i] * np.dot(data[i], x), h )
            f = (1/N) * jfd + (1/2) * Lambda * (np.linalg.norm(x,2)**2) + (1/N) * np.dot(noise,x) + (1/2)*Delta*(np.linalg.norm(x,2)**2)
            return f

        #minimization procedure
        fpriv = minimize(obj_func, x0, method='BFGS').x#empirical risk minimization using scipy.optimize minimize function
        #return fpriv
        return fpriv[:-1], fpriv[-1]

    def svm_non_private(self, data, labels, Lambda, h):
        data = self.prepare_data(data)
        N = len( labels )
        l = len( data[0] )#length of a data point
        x0 = np.zeros(l)#starting point with same length as any data point
        def obj_func(x):
            jfd = self.huber( labels[0] * np.dot(data[0],x), h )
            for i in range(1,N):
                jfd = jfd + self.huber( labels[i] * np.dot(data[i],x), h )
            f = ( 1/N ) *jfd  + (1/2) * Lambda * (np.linalg.norm(x,2)**2)
           
            return f

        #minimization procedure
        f = minimize(obj_func, x0, method='L-BFGS-B').x #empirical risk minimization using scipy.optimize minimize function
        return f[:-1], f[-1]
    
    def decision_function(self, X):
        """ Decision function for softmargin SVM

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        Returns
        -------
        result (ndarray of shape (n_sample,)):
            Decision values    
        """
        return X.dot(self.beta) + self.b
    
    def __margin(self, X, y):
        """ Calcuate margin for softmargin SVM

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample,)):
            labels  

        Returns
        -------
        result (ndarray of shape (n_sample,)):
            Margin values    
        """
        return y * self.decision_function(X)

    def softmargin(self, X, y):
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0

        lr = 1e-3
        for _ in range(5000):
            margin = self.__margin(X, y)

            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - lr * d_beta

            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]

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
        #print((X.detach().numpy()).dot(self.beta.detach().numpy()).shape)
        #print(X.matmul(self.beta) - (X.detach().numpy()).dot(self.beta.detach().numpy()))
        #return X.matmul(self.beta) + self.b
        return X.matmul(self.beta) + self.b

    def soft_huber(self, X, y):
        """ Softmargin with huber loss in pyTorch

         Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample,)):
            labels  

        Returns
        -------
        result (ndarray of shape (n_sample,)):
            Huberloss    
        """
        h = self.huberconst
        result = y * self.torch_decision_function(X)
        result = torch.where(result > 1. + h,
               0.,
               torch.where(result < 1. - h,
                     1 - result,
                     (1 + h - result)**2 / (4 * h)))
        return result    
    
    def softmarginhuber(self, X, y):
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

        lr = 1e-3
        X_torch = torch.tensor(X)
        y_torch = torch.tensor(y)
        optimizer = torch.optim.Adam(params = [self.beta, self.b], lr=lr)
        for _ in range(2000):
            huber = self.soft_huber(X_torch, y_torch)
            #huber /= self.n
            huber.sum().backward()
            #print(self.beta.grad.shape)
            optimizer.zero_grad()
            optimizer.step()
            #with torch.no_grad():
                #update_beta = self.beta.grad + (1/2) * self.lambda_ * ( torch.linalg.norm(self.beta)**2 )
            #    self.beta -= lr * (update_beta)
                #self.beta -= lr * (self.beta.grad)
                
            #    self.b -= lr * self.b.grad
                #self.b.grad = torch.zeros(self.b.grad.shape)
            #    self.beta.grad.zero_()
            #    self.b.grad.zero_()

        self.beta = self.beta.detach().numpy()
        self.b = self.b.detach().numpy()

    def softmarginhuber_clipping(self, X, y):
        """ Softmargin SVM with huberloss

        Parameters
        ----------
        X (ndarray of shape (n_sample, n_feature)):
            features

        y (ndarray of shape (n_sample, )):
            labels   
        """      
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu" 
        device = torch.device(dev)
        cuda = torch.device('cuda')
        cpu= torch.device('cpu')
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.beta = torch.tensor(self.beta, requires_grad = True, device=cpu)
        self.b = torch.tensor([0.0], requires_grad = True, device=cpu)
        #self.beta.to(device)
        #self.b.to(device)

        lr = 1#e-3
        clippingbound = 1.0
        epochs = 3000
        delta = 0.0001
        X_gpu = torch.tensor(X, device=cpu)
        Y_gpu = torch.tensor(y, device=cpu)
        #X_gpu.to(device='cuda')
        #Y_gpu.to(device)
        for _ in range(epochs):
            batchsize = 32
            idx = np.random.randint(self.n, size=batchsize)
            X_torch = X_gpu[idx,:]
            y_torch = Y_gpu[idx]
            #print(X_torch.device)
            #X_torch.to(device)
            #y_torch.to(device)
            huber = self.soft_huber(X_torch, y_torch)
            huber.sum().backward()
            #print(self.beta.grad.shape)
            with torch.no_grad():
                update = lr * self.beta.grad
                if torch.linalg.norm(update) > 1.0:
                    update /= torch.linalg.norm(update)
                sigma = ((batchsize/self.n) * (math.sqrt(epochs*21*np.log(1/delta))) * self.C)/self.epsilon
                tensor_mean = torch.mean(update)
                rv = multivariate_normal(mean=tensor_mean.item(), cov=1*sigma)
                random_elements = rv.rvs(size=self.d)
                random_tensor = torch.tensor(random_elements, device=cpu)
                #random_tensor.to(device)
                self.beta -= (update + random_tensor)
                
                self.b -= lr * self.b.grad
                #self.b.grad = torch.zeros(self.b.grad.shape)
                self.beta.grad.zero_()
                self.b.grad.zero_()

        self.beta = self.beta.detach().numpy()
        self.b = self.b.detach().numpy()
    
    def softmarginhuber_with_output(self, X, y):
        """ Softmargin SVM with huberloss and output perturbation

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
        scale = len(y) * self.lambda_ * self.epsilon / 2

        lr = 1e-3
        X_torch = torch.tensor(X)
        y_torch = torch.tensor(y)
        for _ in range(3000):
            huber = self.soft_huber(X_torch, y_torch)
            huber /= self.n
            huber.sum().backward()
            #print(self.beta.grad.shape)
            with torch.no_grad():
                update_beta = self.beta.grad + (1/2) * self.lambda_ * ( torch.linalg.norm(self.beta)**2 )
                self.beta -= lr * (update_beta)
                
                self.b -= lr * self.b.grad
                #self.b.grad = torch.zeros(self.b.grad.shape)
                self.beta.grad.zero_()
                self.b.grad.zero_()

        self.beta = self.beta.detach().numpy()
        l = len( X[0] )
        self.beta += self.noisevector(l, scale)
        self.b = self.b.detach().numpy()

    def softmarginhuber_with_objective(self, X, y):
        """ Softmargin SVM with huberloss and output perturbation

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
        l = len( X[0] )
        N = len( y )
        c = 1 / (2 * self.huberconst)
        tmp = c / (N * self.lambda_)
        Epsilonp = self.epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
        Delta = 0.0
        if Epsilonp > 0:
            Delta = 0
        else:
            Delta = c / ( N * (np.exp(self.epsilon/4)-1) ) - self.lambda_
            Epsilonp = self.epsilon / 2

        noise = self.noisevector(l, Epsilonp/2)
        noise_torch = torch.tensor(noise)
        lr = 1e-3
        X_torch = torch.tensor(X)
        y_torch = torch.tensor(y)
        for _ in range(10000):
            huber = self.soft_huber(X_torch, y_torch)
            huber /= N
            huber.sum().backward()
            #print(self.beta.grad.shape)
            with torch.no_grad():
                update_beta = self.beta.grad + (1/N) * torch.dot(noise_torch,self.beta) + 0.5*Delta*(torch.linalg.norm(self.beta)**2)
                self.beta -= lr * (update_beta)
                
                self.b -= lr * self.b.grad
                #self.b.grad = torch.zeros(self.b.grad.shape)
                self.beta.grad.zero_()
                self.b.grad.zero_()

        self.beta = self.beta.detach().numpy()
        self.b = self.b.detach().numpy()

    def softmargin_with_noise(self, X, y):
        """ Softmargin SVM with output perturbation

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
       
        lr = 1e-3
        for _ in range(500):
            margin = self.__margin(X, y)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            d_beta = d_beta + self.lambda_ * (np.linalg.norm(d_beta,2)**2)
            self.beta = self.beta - lr * d_beta
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

        scale = len(y) * self.lambda_ * self.epsilon / 2
        l = len( X[0] )
        self.beta = self.beta + self.noisevector(l, scale)
        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]
    
    def softmargin_with_obj(self, X, y):
        """ Softmargin SVM with objective perturbation

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
        c = 0.0
        tmp = c / (N * self.lambda_)
        Epsilonp = self.epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
        Delta = 0.0
        if Epsilonp > 0:
            Delta = 0
        else:
                Delta = c / ( N * (np.exp(self.epsilon/4)-1) ) - self.lambda_
                Epsilonp = self.epsilon / 2
        lr = 1e-3
        for _ in range(5000):
            margin = self.__margin(X, y)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            #d_beta = d_beta + self.lambda_ * (np.linalg.norm(d_beta,2)**2)
            
            noise = self.noisevector(l, Epsilonp/2)
            d_beta = d_beta + (1/N) * np.dot(noise,d_beta) + 0.5*Delta*(np.linalg.norm(d_beta,2)**2)
            self.beta = self.beta - lr * d_beta
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b
            
        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]

    def softmargin_clipping(self, X, y):
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
        l = len( X[0] )
        T = 10000
        delta = 1 / (self.n ** 2)
        sigma = math.sqrt((T * np.log(1.26 / delta)) / (self.epsilon ** 2))
        covmatrix = np.identity(self.d) * sigma
        lr = 1e-3
        for t in range(T):
            margin = self.__margin(X, y)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            if np.linalg.norm(d_beta) > 1.0:
                d_beta /= np.linalg.norm(d_beta)
            noise = multivariate_normal.rvs(cov=covmatrix, size=1)
            self.beta = self.beta - lr * (d_beta + noise)
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]
    
    def fit(self, x, y):
        """trains a svm on a given dataset x.

        Parameters
        ----------
        x : (ndarray of shape (n_sample, n_feature))
            Features

        y : (ndarray of shape (n_sample,))
            Labels
        """
        if self.alg != 'softmargin_clipping':
            for i in range(x.shape[0]):
                if np.linalg.norm(x[i,:]) > 1.0:
                    x[i,:] /= np.linalg.norm(x[i,:])
        XY = x * y[:, None]
        num = x.shape[0]
        dim = x.shape[1]
        if self.alg == 'output_pertubation':
            self.weights = self.train_svm_outputperturb(XY, num, dim, self.lambda_, self.epsilon, self.huberconst)
        elif self.alg == 'objective_pertubation':
            self.weights = self.train_svm_objectiveperturb(XY, num, dim, self.lambda_, self.epsilon, self.huberconst)
        elif self.alg == 'non_private':
            self.weights = self.train_svm_nonpriv(XY, num, dim, self.lambda_, self.huberconst)
        elif self.alg == 'output':
            self.weights, self.bias = self.svm_output_train(x, y, self.epsilon, self.lambda_, self.huberconst)
        elif self.alg == 'obj':
            self.weights, self.bias = self.svm_objective_train(x, y, self.epsilon, self.lambda_, self.huberconst)        
        elif self.alg == 'non':
            self.weights, self.bias = self.svm_non_private(x, y, self.lambda_, self.huberconst)
        elif self.alg == 'softmargin':
            self.softmargin(x, y)
        elif self.alg == 'softmargin_objective':
            self.softmarginhuber_with_objective(x,y)
        elif self.alg == 'softmargin_output':
            self.softmarginhuber_with_output(x,y)
        elif self.alg == 'softmargin_clipping':
            self.softmarginhuber_clipping(x,y)
        elif self.alg == 'softmargin_huber':
            self.softmarginhuber(x,y)
        else:
            print('This is not an available algorithm')    

    def predict(self, data):
        """Returns decision values for data.
        
        Parameters
        ----------
        data : (ndarray of shape (n_sample, n_feature))
              Features

        Returns
        -------
        vote : (ndarray of shape (n_sample,))
              Decision
        """
        #data = normalize(X=data, norm='l1', axis=1)
        if "softmargin" in self.alg:
            return self.decision_function(data)
        else:
            return self.decision_svmhuber(self.bias, data)
    

class MultiSVM:
    """ Multiclass SVM with privacy

    Parameters
    ----------
    alg : string (default='objective_pertubation')
          Chooses the privacy algorithm. Either 'non_private' for no privacy, 'output_pertubation' for
           privacy or 'objective_pertubation' for privacy

    C : float (default=1.0)
          Regularization parameter. Must be greater than 0.0       

    lambda_ : float (default=0.5)
          Regularization parameter

    epsilion : float (default=0.5)
          Privacy parameter

    huberconst : float (default=0.5)
          Constant value for huber loss      

    decision_function_shape : String (default='ovo')
          Shape of decision function

    train_shape : String (default='ovo')
          Multiclass training option. Default one vs one, can be set to one vs rest (not recommended)                  
    """
    def __init__(self, alg='objective_pertubation', C= 1.0, lambda_ = 0.5, epsilon = 0.5, huberconst = 0.5, decision_function_shape='ovo', train_shape='ovo'):
        self.alg, self.C, self.lambda_ = alg, C, lambda_
        self.epsilon, self.huberconst = epsilon, huberconst
        self.decision_function_shape = decision_function_shape
        self.train_shape = train_shape
        self.classifiers = []
        self.pls = None
        self.classes = 0

    def fit(self, x, y):
        """trains a svm on a given dataset x.

        Parameters
        ----------
        x : (ndarray of shape (n_sample, n_feature))
            Features

        y : (ndarray of shape (n_sample,))
            Labels
        """
        
        labels = np.unique(y)
        self.classes = labels.astype(int)
        self.n_class = len(labels)
        if self.train_shape == 'ovr':  # one-vs-rest method
            for label in labels:
                y1 = np.array(y)
                y1[y1 != label] = -1.0
                y1[y1 == label] = 1.0
                clf = BinarySVM(alg= self.alg, C= self.C, lambda_= self.lambda_, epsilon= self.epsilon, huberconst= self.huberconst)
                clf.fit(x, y1)
                self.classifiers.append(copy.deepcopy(clf))
        else:
            start = time.time()
            for i in range(self.n_class):
                for j in range(i+1, self.n_class):
                    print(i,j)
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    x1, y1 = np.r_[x[neg_id], x[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.0
                    y1[y1 == labels[j]] = 1.0
                    clf = BinarySVM(alg= self.alg, C= self.C, lambda_= self.lambda_, epsilon= self.epsilon, huberconst= self.huberconst)
                    clf.fit(x1, y1)
                    self.classifiers.append(copy.deepcopy(clf))
            print(time.time() - start)

    def decision_function(self, data):
        """ Returns decision function results

        Parameters
        ----------
        data : (ndarray of shape (n_sample, n_feature))
              Features

        Returns
        -------
        score : (ndarray of shape (n_sample,n_classifier))
              Decision function
        """
        n_samples = data.shape[0]
        score = np.zeros((n_samples, len(self.classifiers)))
        for i in range(len(self.classifiers)):
                score[:,i] = self.classifiers[i].predict(data)
        if self.decision_function_shape == 'ovr':
            print(self.classes)
            return self._ovr_decision_function(score > 0, score, len(self.classes))
        return score
    
    def _ovr_decision_function(self, predictions, confidences, n_classes):
        """Compute a continuous, tie-breaking OvR decision function from OvO.
        It is important to include a continuous value, not only votes,
        to make computing AUC or calibration meaningful.
        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Predicted classes for each binary classifier.
        confidences : array-like, shape (n_samples, n_classifiers)
            Decision functions or predicted probabilities for positive class
            for each binary classifier.
        n_classes : int
            Number of classes. n_classifiers must be
            ``n_classes * (n_classes - 1 ) / 2``
        """
        n_samples = predictions.shape[0]
        print(n_classes)
        votes = np.zeros((n_samples, n_classes))
        sum_of_confidences = np.zeros((n_samples, n_classes))
    
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                sum_of_confidences[:, i] -= confidences[:, k]
                sum_of_confidences[:, j] += confidences[:, k]
                votes[predictions[:, k] == 0, i] += 1
                votes[predictions[:, k] == 1, j] += 1
                k += 1

        # Monotonically transform the sum_of_confidences to (-1/3, 1/3)
        # and add it with votes. The monotonic transformation  is
        # f: x -> x / (3 * (|x| + 1)), it uses 1/3 instead of 1/2
        # to ensure that we won't reach the limits and change vote order.
        # The motivation is to use confidence levels as a way to break ties in
        # the votes without switching any decision made based on a difference
        # of 1 vote.
        transformed_confidences = (sum_of_confidences /
                                (3 * (np.abs(sum_of_confidences) + 1)))
        return votes + transformed_confidences

    
    def predict(self, data):
        """Returns decision values for data.
        
        Parameters
        ----------
        data : (ndarray of shape (n_sample, n_feature))
              Features

        Returns
        -------
        vote : (ndarray of shape (n_sample,))
              Decision
        """
        
        n_samples = data.shape[0]
        if self.decision_function_shape == 'ovr':
            score = self.decision_function(data)
            return np.argmax(score, axis=1)
        else:
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i+1, self.n_class):
                    res = self.classifiers[clf_id].predict(data)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    # print i, j
                    # print 'res = ', res
                    # print 'vote = ', vote
                    clf_id += 1
            return np.argmax(vote, axis=1)

    def get_weight(self):
        """ Returns weight of SVM

        Returns
        -------
        weight : (ndarray of shape ((self.n_class * (self.n_class - 1) / 2), n_features))
        """
        weights = self.classifiers[0].beta
        for i in range(1, len(self.classifiers)):
            weights = np.vstack([weights, self.classifiers[i].beta])
        return weights    

    def get_classes(self):
        """ Get the class labels

        Returns
        -------
        classes : (ndarray of shape(n_classes,))
        """
        return self.classes    

