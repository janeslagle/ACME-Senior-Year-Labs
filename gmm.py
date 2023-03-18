# solutions.py
"""Volume 3: Gaussian Mixture Models. Solutions File."""

import numpy as np
from scipy import stats as st
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import time
import sklearn
from sklearn import mixture

from sklearn.metrics import confusion_matrix

class GMM:
    # Problem 1
    def __init__(self, n_components, weights=None, means=None, covars=None):
        """
        Initializes a GMM.
        
        The parameters weights, means, and covars are optional. If fit() is called,
        they will be automatically initialized from the data.
        
        If specified, the parameters should have the following shapes, where d is
        the dimension of the GMM:
            weights: (n_components,)
            means: (n_components, d)
            covars: (n_components, d, d)
        """
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.covars = covars
        
    # Problem 2
    def component_logpdf(self, k, z):
        """
        Returns the logarithm of the component pdf. This is used in several computations
        in other functions.
        
        Parameters:
            k (int) - the index of the component
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the log pdf of the component at 
        """
        #get equation given in lab manual:
        wk = self.weights[k]  #want kth weight w_k
        logN = np.log(st.multivariate_normal.pdf(z, self.means[k], self.covars[k])) #want kth in the normal log 
        
        return np.log(wk) + logN
        
    # Problem 2
    def pdf(self, z):
        """
        Returns the probability density of the GMM at the given point or points.
        
        Parameters:
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the GMM pdf at z
        """
        #have all of the logs from component_logpdf method:
        logpdfs = [self.component_logpdf(k, z) for k in range(self.n_components)]  
        logpdfs = np.array(logpdfs)
        
        #then take logs to get rid of logs, sum them bc doing P(z|theta) formula given in lab manual
        return np.sum(np.exp(logpdfs)) 
    
    # Problem 3
    def draw(self, n):
        """
        Draws n points from the GMM.
        
        Parameters:
            n (int) - the number of points to draw
        Returns:
            ((n,d) ndarray) - the drawn points, where d is the dimension of the GMM.
        """
        #draw sample of 10,000 pts from GMM defined in prob 2:
        draws = []  #list to store all of the draws in
        for _ in range(n):
            X = np.random.choice(a = self.n_components, p = self.weights)
            Z = st.multivariate_normal.rvs(mean = self.means[X], cov = self.covars[X])
            
            draws.append(Z)
            
        return np.array(draws)
            
    # Problem 4
    def _compute_e_step(self, Z):
        """
        Computes the values of q_i^t(k) for the given data and current parameters.
        
        Parameters:
            Z ((n, d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components, n) ndarray): an array of the computed q_i^t(k) values, such
                    that result[k,i] = q_i^t(k).
        """
        l_ik = np.array([self.component_logpdf(k, Z) for k in range(self.n_components)]).T
        Li = np.max(l_ik, axis=1)
        
        numerator = np.exp(l_ik.T - Li)
        denominator = np.sum(numerator.T, axis = 1)
            
        qs = (numerator/denominator)
        
        return qs
        
    # Problem 5
    def _compute_m_step(self, Z):
        """
        Takes a step of the expectation maximization (EM) algorithm. Return
        the updated parameters.
        
        Parameters:
            Z (n,d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components,) ndarray): the updated component weights
            ((n_components,d) ndarray): the updated component means
            ((n_components,d,d) ndarray): the updated component covariance matrices
        """
        n = Z.shape[0]    #told in docstring for prob that Z has shape (n, d)
        d = Z.shape[1]
        
        #get q_it from prob 4:
        q = self._compute_e_step(Z)
        
        #calculate the new weights:
        new_weights = []
        for k in range(self.n_components):
            new_weights.append((1/n)*np.sum(q[k]))
            
        #calculate the new means:
        new_means = q@Z/np.sum(q, axis=1).reshape(-1,1)
        
        #calculate the new covariance matrices: TAs gave us this code
        centered = np.expand_dims(Z, 0) - np.expand_dims(new_means, 1)
        
        new_covars = np.einsum("Kn, Knd, KnD -> KdD", q, centered, centered)/np.sum(q, axis=1).reshape(-1,1,1)
        
        return new_weights, new_means, new_covars
        
    # Problem 6
    def fit(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model by applying the Expectation Maximization algorithm until the
        parameters appear to converge.
        
        Parameters:
            Z ((n,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            self
        """
        n, d = Z.shape[0], Z.shape[1]
        
        #check if all parameters are None, then initialize them as want:
        
        #initialize means by randomly selecting pts from dataset:
        if self.means is None:
            means = []
            for i in range(self.n_components):
                means.append(Z[np.random.randint(0,n)])  #randomly select pts from dataset
            self.means = means
        
        #initialize random covariances as diagonal matrices based on variance of data
        if self.covars is None:
            covars = []
            for i in range(self.n_components):
                covars.append(np.diag(np.var(Z, axis=0)))
            self.covars = covars
            
        #get random weights, make sure they add up to 1:
        if self.weights is None:
            self.weights = np.ones(self.n_components)/self.n_components
            
        #NOW perform expectation maximization algorithm:
        old_weights = self.weights
        old_means = self.means
        old_covars = self.covars

        #use funcs from probs 4, 5 to calc parameters @ each step:
        for i in range(maxiter):
            #use previous probs to update:
            new_weights, new_means, new_covars = self._compute_m_step(Z)
            
            #want measure the change in parameters w/ each iteration using code given in lab manual:
            change = (np.max(np.abs(new_weights - old_weights))
                    + np.max(np.abs(new_means - old_means))
                    + np.max(np.abs(new_covars - old_covars)))
                    
            #update the weights, means, covars to be new ones got:
            old_weights = np.array(new_weights)
            old_means = np.array(new_means)
            old_covars = np.array(new_covars)
            
            #want repeat until parameters converge:
            if change < tol:
                break        #so finish if converged
                
            self.weights = old_weights
            self.means = old_means
            self.covars = old_covars
        
        return self
        
    # Problem 8
    def predict(self, Z):
        """
        Predicts the labels of data points using the trained component parameters.
        
        Parameters:
            Z ((m,d) ndarray): the data to label; d is the dimension of the data.
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        #given set of data pts, return which cluster has highest pdf density for each data pt
        #so want to do the cluster(z) = argmax(blah blah blah) func given in lab manual, that's what returning here
        
        return np.argmax([np.exp(self.component_logpdf(k,Z)) for k in range(self.n_components)], axis=0)
        
    def fit_predict(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model and predicts cluster labels.
        
        Parameters:
            Z ((m,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        return self.fit(Z, tol, maxiter).predict(Z)

# Problem 3
def problem3():
    """
    Draw a sample of 10,000 points from the GMM defined in the lab pdf. Plot a heatmap
    of the pdf of the GMM (using plt.pcolormesh) and a hexbin plot of the drawn points.
    How do the plots compare?
    """
    #plot pdf of GMM:
    gmm = GMM(n_components = 2,
            weights = np.array([0.6, 0.4]),
            means = np.array([[-0.5, -4.0], [0.5, 0.5]]),
            covars = np.array([[[1, 0],[0, 1]],
            [[0.25, -1],[-1, 8]],
            ]))
    
    ## Create the grid to plot on
    x = np.linspace(-8,8,100) 
    y = np.linspace(-8,8,100)
    X, Y = np.meshgrid(x, y)
    ## Calculate the pdf at each point
    # If your pdf function uses array broadcasting, you can do the following:
    Z = gmm.pdf(np.dstack((X,Y)))
    # Otherwise, you need to iterate over each point:
    Z = np.array([[
            gmm.pdf([X[i,j], Y[i,j]]) for j in range(100)
            ] for i in range(100)
            ])
    ## Create the plot
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.show()
    
    #plot hexbin plot of drawn points:
    draws = gmm.draw(10000)  #get the draws using prob 3, want draw from same gmm defined in prob 2
    plt.hexbin(draws[:,0], draws[:,1])
    plt.show()
    
# Problem 7
def problem7(filename='problem7.npy'):
    """
    The file problem6.npy contains a collection of data drawn from a GMM.
    Train a GMM on this data with n_components=3. Plot the pdf of your
    trained GMM, as well as a hexbin plot of the data.
    """
    data = np.load(filename)  #load in file
    
    #create GMM thing using GMM class:
    gmm_ting = GMM(n_components = 3)
    gmm_ting.fit(data)        #use prob 6 to fit that data!
    
    #told to plot on range of [-4,4] x [-4,4]
    x = np.linspace(-4,4,100)
    y = np.linspace(-4,4,100)
    X,Y = np.meshgrid(x, y)
    
    #want plot pdf of trained GMM so find pdf @ each pt, need to iterate over each point to do so
    Z = np.array([[gmm_ting.pdf([X[i,j], Y[i,j]]) for j in range(100)] for i in range(100)])
    
    #plot the pdf:
    plt.pcolormesh(X, Y, Z, shading = "auto")
    plt.title("PDF of Trained GMM")
    
    plt.show()
    
    #plot the hexbin: need draw from gmm object to do so
    draws = gmm_ting.draw(10000)
    plt.title("Draws of Training Data")
    plt.hexbin(draws[:,0], draws[:,1], gridsize = (30,30))
    plt.xlim((-4,4))
    plt.ylim((-4,4))
    
    plt.show()
    
# Problem 8
def get_accuracy(pred_y, true_y):
    """
    Helper function to calculate the actually clustering accuracy, accounting for
    the possibility that labels are permuted.
    
    This computes the confusion matrix and uses scipy's implementation of the Hungarian
    Algorithm (linear_sum_assignment) to find the best combination, which is generally
    much faster than directly checking the permutations.
    """
    # Compute confusion matrix
    cm = confusion_matrix(pred_y, true_y)
    # Find the arrangement that maximizes the score
    r_ind, c_ind = linear_sum_assignment(cm, maximize=True)
    return np.sum(cm[r_ind, c_ind]) / np.sum(cm)
    
def problem8(filename='classification.npz'):
    """
    The file classification.npz contains a set of 3-dimensional data points "X" and 
    their labels "y". Use your class with n_components=4 to cluster the data.
    Plot the points with the predicted and actual labels, and compute and return
    your model's accuracy. Be sure to check for permuted labels.
    
    Returns:
        (float) - the GMM's accuracy on the dataset
    """
    #get the X, Y data pts. from file, read them in from file:
    with np.load(filename) as data:
        X = data['X']
        y = data['y']
        
    #create the GMM class object w/ n_components = 4:
    gmm = GMM(n_components = 4)
    y_pred = gmm.fit_predict(X)   #want plot predicted labels, so get predicted here
      
    #graph labels of OG data:
    #loop through points, plot with labels assigned using OG y
    fig = plt.figure()
    colors = ['deeppink', 'cornflowerblue', 'forestgreen', 'orange']
    ax = fig.add_subplot(122, projection='3d')
    for label in range(4):
        mask = []
        for i, val in enumerate(y):
            if val == label:
                mask.append(i)
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=colors[label])
    plt.title("original data clustering")
    plt.show()
        
    fig = plt.figure()
    #graph labels of predicted data:
    #loop through points and plot with labels assigned using pred_y now instead of OG y have
    ax = fig.add_subplot(122, projection='3d')
    for label in range(4):
        mask = []
        for i, val in enumerate(y_pred):
            if val == label:
                mask.append(i)
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=colors[label])
    plt.title('predicted clustering')
    plt.show()

    return get_accuracy(y_pred, y)  #want return GMM's accuracy on dataset

# Problem 9
def problem9(filename='classification.npz'):
    """
    Again using classification.npz, compare your class, sklearn's GMM implementation, 
    and sklearn's K-means implementation for speed of training and for accuracy of 
    the resulting clusters. Print your results. Be sure to check for permuted labels.
    """
    #load in data from file have:
    with np.load(filename) as data:
        X = data['X']
        y = data['y']
        
    #use sklearn GMM implementation thing:
    sk_gmm = mixture.GaussianMixture(4, max_iter = 200)
    start_time = time.time()   #want to time it
    predicts = sk_gmm.fit_predict(X)
    end_time = time.time() - start_time
    
    print("sklearn GMM Accuracy: ", get_accuracy(predicts, y))
    print("sklearn GMM Time: ", end_time)
    print("")
    
    #use sklearn k-means implementation thing:
    sk_kmeans = sklearn.cluster.KMeans(4, max_iter = 200, tol = 1e-3)
    start_time = time.time()
    predicts = sk_kmeans.fit_predict(X)
    end_time = time.time() - start_time
    print("sklearn K-means Accuracy: ", get_accuracy(predicts, y))
    print("sklearn K-means Time: ", end_time)
    
    #now use our gmm method:
    gmm = GMM(4)
    start_time = time.time()
    predicts = gmm.fit_predict(X)
    end_time = time.time() - start_time
    
    print("")
    print("My very own oh la la GMM Accuracy: ", get_accuracy(predicts, y))
    print("My over own oh la la GMM Time: ", end_time)
    
if __name__ == '__main__':
    #test prob 2:
    """gmm = GMM(n_components = 2,
            weights = np.array([0.6, 0.4]),
            means = np.array([[-0.5, -4.0], [0.5, 0.5]]),
            covars = np.array([[[1, 0],[0, 1]],
            [[0.25, -1],[-1, 8]],
            ]))"""
    
    #print(gmm.pdf(np.array([1.0, -3.5])))
    #print(gmm.component_logpdf(0, np.array([1.0, -3.5])))
    #print(gmm.component_logpdf(1, np.array([1.0, -3.5])))
    
    #test prob 3:
    #print(problem3())
    
    #test prob 4:
    data = np.array([
        [0.5, 1.0],
        [1.0, 0.5],
        [-2.0, 0.7]
    ])
    
    #print(gmm._compute_e_step(data))
    
    #test prob 5:
    #print(gmm._compute_m_step(data))
    
    #test probs 6 and 7:
    #print(problem7(filename='problem7.npy'))
    
    #test prob 8:
    #print(problem8(filename='classification.npz'))
    
    #test prob 9:
    print(problem9(filename='classification.npz'))
        
    pass
    
