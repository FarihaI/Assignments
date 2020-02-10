import numpy as np
import math
import script_regression as error
import MLCourse.utilities as utils
import matplotlib.pyplot as plt



### Fariha Imam ###


# -------------
# - Baselines -
# -------------

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.0,
            'features': range(385),
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        #self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples
        # with all the features, inorder for FSlinear regression work we change np.linalg.inv to np.linalg.pinv
        self.weights = np.linalg.pinv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------


##### 2(c) #######
class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({
            'regwgt': 0.01,
            'features': range(385),
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.pinv(inner).dot(Xless.T).dot(ytrain) / numsamples


    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

######### 2(d)######

class LassoRegression(Regressor):
  
    def __init__( self, parameters={} ):
        self.params = utils.update_dictionary_items({'regwgt': 0.01, 'features': range(385),
                                                     'tolerance' :10e-4,
        }, parameters)
        
        
    def prox(self, w, stepsize,regwgt):
        
        np.where(w > stepsize*regwgt)
        w-= stepsize*regwgt
        np.where(w < -stepsize*regwgt)
        w += stepsize*regwgt
        return w
                
               
    def learn(self, Xtrain, ytrain):
        
        n = Xtrain.shape[0] 
        regwgt=self.params['regwgt']
        self.weights = np.zeros(Xtrain.shape[1])   
        err =np.Infinity 
        tolerance = self.params['tolerance']  
        XX =Xtrain.T.dot(Xtrain) /n
        Xy = Xtrain.T.dot(ytrain)/n
        stepsize =  1/(2*np.linalg.norm(XX))
        
        cw= error.geterror(Xtrain.dot(self.weights), ytrain) 

        while abs(cw-err) > tolerance:
            err = cw
            term=np.subtract(self.weights, stepsize*np.dot(XX, self.weights))
            termn=np.add(term,stepsize*Xy)
            
            self.weights=self.prox(termn, stepsize,regwgt) 
            cw = error.geterror(Xtrain.dot(self.weights), ytrain)
        
         
  
  ######### 2(e)##########
 

class stochasticGD(Regressor):
  
     def __init__( self, parameters={} ):
         self.params = utils.update_dictionary_items({'regwgt': 0.01, 'features': range(385),
                                                      "epochs": 1000,"stepsize":0.01,}, parameters)
         self.noofruns = 5
         self.error = np.zeros(1000)
         
     def learn(self, Xtrain, ytrain):
        n = Xtrain.shape[0]
       
        self.weights = np.random.rand(Xtrain.shape[1])* self.params['regwgt']
        no =self.params["stepsize"]
       
        for i in range (self.params["epochs"]):
            for j in range(n):
                XjW = (Xtrain[j].T.dot(self.weights))
                gradient = (XjW - ytrain[j]) * Xtrain[j]
                nt = no/(1+i) 
                self.weights = np.subtract(self.weights, nt*gradient)
                
            self.error[i] += error.geterror(np.dot(Xtrain, self.weights), ytrain)
           
        
     def forplot(self):   
        return self.error/self.noofruns

     def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest  
     
       
 
    
    
    
    ######### 2(f)##########

class BatchGradientDescent(Regressor):
   
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'regwgt': 0.01, 'stepsize':0.01,'tolerance':10e-4,
        'maxiteration':1000}, parameters)
   
    def lsearch(self, Xtrain, ytrain, weight, gradient, cw):
    
        t = 0.5 
        tolerance = self.params['tolerance']
        stepsize = 1.0
        obj = cw
        maxinteration = self.params['maxiteration']
        iteration = 0
        for iteration in range(maxinteration): 
            weight = self.weights - stepsize * gradient
            if (cw < obj-tolerance):break
            else: stepsize = t * stepsize
            cw = error.geterror(np.dot(Xtrain, weight), ytrain)
            iteration = iteration + 1
        if iteration == maxinteration:
            stepsize = 0
            return stepsize
        return stepsize
    
    
    
    def learn(self, Xtrain, ytrain):
        n = Xtrain.shape[0]
        self.weights = np.random.rand(Xtrain.shape[1])* self.params['regwgt'] 
        err = np.Infinity
        tolerance = self.params['tolerance']
        stepsize = self.params['stepsize']
       
        cw= error.geterror(Xtrain.dot(self.weights), ytrain) 
       
        while abs(cw-err) > tolerance:
            err = cw
            term = np.subtract((np.dot(Xtrain, self.weights)),ytrain)
            gradient = np.dot(Xtrain.T, term)/n
            stepsize = self.lsearch(Xtrain, ytrain, self.weights, gradient, cw)   
            self.weights = self.weights - stepsize*gradient
            cw = error.geterror(np.dot(Xtrain, self.weights), ytrain)
        
            
            
def predict(self, Xtest):
       
        ytest = np.dot(Xtest, self.weights)
        return ytest 
    
    
    
                                 ######## Bonus Question########
 
    
    
######### Bonus 5(a) ##########

class MomentumLinearRegression(Regressor):
   
    def __init__(self, parameters = {}):
         self.params = utils.update_dictionary_items({"iteration": 1000,}, parameters)
    def learn(self, Xtrain, ytrain):
         self.weights = np.random.randn(Xtrain.shape[1])
         self.noofruns = 5
         self.error = np.zeros(1000)
         n = Xtrain.shape[0]
         stepsize=0.01
         beta=0.9
         v=0    
         t=0
         
         for t in range (self.params["iteration"]):
            for j in range(n):
                XjW = (Xtrain[j].T.dot(self.weights))
                gradient = (XjW - ytrain[j]) * Xtrain[j]
                vt= beta * v + (1 - beta) * gradient
                #With out bias correction, we will be using vt directly in self.weight 
                #self.weights = self.weights - stepsize * vt   
                #Bias corrected
                vhat=vt/(1-beta)
                self.weights = self.weights - stepsize * vhat                
                self.error[t] += error.geterror(np.dot(Xtrain, self.weights), ytrain)
               
   
    def predict(self, Xtest):
       
        ytest = np.dot(Xtest, self.weights)
        return ytest 
 
                                 
########## Bonus 5(b)########       
#whereas Adam updates are directly estimated using a running average of first and second moment of the gradien
    
class ADAMLinearRegression(Regressor):
   
    def __init__(self, parameters = {}):
         self.params = utils.update_dictionary_items({"iteration":1000 ,}, parameters)
     
    
    def learn(self, Xtrain, ytrain):
         self.weights = np.random.randn(Xtrain.shape[1])
         self.noofruns = 5
         self.error = np.zeros(1000)
         n = Xtrain.shape[0]
         stepsize=0.01
         beta1=0.9
         beta2=0.999
         m=0
         v=0    
         t=0
         
         for t in range (self.params["iteration"]):
            for j in range(n):
                XjW = (Xtrain[j].T.dot(self.weights))
                gradient = (XjW - ytrain[j]) * Xtrain[j]
                mt = beta1 * m + (1 - beta1) * gradient
                vt = beta2 * v + (1 - beta2) * np.power(gradient, 2)
                #Without bias correction we will directly be using mt and vt in self.weight 
                #self.weights = self.weights - stepsize * mt / (np.sqrt(vt) + 10e-8) 
                #Bias corrected
                mhat=mt/(1-beta1)
                vhat=vt/(1-beta2)
                self.weights = self.weights - stepsize * mhat / (np.sqrt(vhat) + 10e-8)                
                self.error[t] += error.geterror(np.dot(Xtrain, self.weights), ytrain)
                
    
    def predict(self, Xtest):
       
        ytest = np.dot(Xtest, self.weights)
        return ytest 
    
 ####### Bonus 5(C) ######
    
#INITIALIZATION BIAS CORRECTION
# The moving averages initialized as (vectors of) 0’s, leading to moment estimates that are biased towards zero 
#  We therefore divide by (1 − β^t(2))to correct the initialization bias.
# bias is corrected in the momentum and adam class
    
