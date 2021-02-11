import numpy as np
import math
from sklearn.svm import SVC

class private_SVC:
    """ Multiclass SVC with privacy based on one vs one classification

    Parameters
    ----------
    C : float, optional (default=1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel : string, optional (default=linear)
        As of now, only linear SVC is supported
    
    lambda_ : float, optional (default=0.5)
          Regularization parameter

    epsilion : float, optional (default=0.5)
          Privacy parameter         
    """
    def __init__(self, C= 1.0, kernel='linear', epsilon= 1.0, lambda_= 0.5):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.weights = 0.0
        self.noise = 0.0
        self.n_classes = 0
        self.dim = 0
        self.beta = 0.0
        self.clf = SVC(C= self.C, kernel= self.kernel, decision_function_shape='ovo')

    def noisevector(self, rate_lambda):
        """Generates a noise vector following Laplace distribution.

        The distribution of norm is Erlang distribution with parameters (dim, 
        rate_lambda). For the direction, pick uniformly by sampling dim number of 
        i.i.d. Gaussians and normalizing them.

        Args:
            rate_lambda (float): epsion^(-rate_lambda*x) in Erlang distribution.
        
        Returns:
            res (ndarray of shape (dim,)): Noise vector.
        
        References:
            https://ergodicity.net/2013/03/21/generating-vector-valued-noise-for-differential-privacy/
        """
        # generate norm, after Numpy version 1.17.0
        normn = np.random.default_rng().gamma(self.dim, 1 / rate_lambda, 1)
        # generate direction
        r1 = np.random.normal(0, 1, self.dim)
        n1 = np.linalg.norm(r1, 2)  # get the norm of r1
        r2 = r1 / n1  # normalize r1
        # get the result noise vector
        res = r2 * normn
        return res
    
    def vote(self, X):
        """Internally function to vote in the ovo multiclass case

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_classifiers)
            Prediction of each binary classifier.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Most voted classes in X.
        """ 
        votes = np.zeros((X.shape[0], self.n_classes))
        for k in range(0, X.shape[0]):
            pointer = 0
            for i in range(0, self.n_classes):
                for j in range(i + 1, self.n_classes):
                    if(X[k, pointer]) > 0.0:
                        votes[k, i] += 1
                    else:
                        votes[k, j] += 1
                    pointer += 1

        return np.argmax(votes, axis=1)

    def generate_noise(self):
        """
        Internally function to generate the noise matrix

        Returns
        ------
        noise_matrix : array, shape (n_samples, n_clf)
            Noise matrix for weight matrix
        """
        noise_matrix = np.zeros((self.dim, self.clf.coef_.shape[0]))
        scale = (2 * self.C * math.sqrt(2* np.log(1.25/0.0001)) * math.sqrt(21)) / self.epsilon
        
        for i in range(self.clf.coef_.shape[0]):
            noise = np.random.default_rng().normal(scale= scale, size = self.dim)
            noise_matrix[:, i] = noise#self.noisevector(self.beta)
        return noise_matrix    

    
    def fit(self, X, Y):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)
        """    
        self.clf.fit(X, Y)
        self.dim = X.shape[1]
        num = X.shape[0]
        self.beta = num * self.lambda_ * self.epsilon / 2
        self.noise = self.generate_noise()
        self.n_classes = np.amax(Y).astype(int) + 1

    def predict(self, X):
        """Perform classification on samples in X with fixed random noise.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        print(self.clf.coef_.T)
        print(self.noise)
        prediction = np.dot(X, (self.clf.coef_.T + self.noise)) + self.clf.intercept_
        return self.vote(prediction)


    def predict_with_random_noise(self, X):
        """Perform classification on samples in X with newly calculated randomnoise.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        random_noise = self.generate_noise()
        prediction = np.dot(X, (self.clf.coef_.T + random_noise)) + self.clf.intercept_
        return self.vote(prediction)

    def get_weights(self):
        return self.weights