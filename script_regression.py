import csv
import random
import math
import numpy as np

import regressionalgorithms as algs
import MLCourse.dataloader as dtl
import MLCourse.plotfcns as plotfcns
import matplotlib.pyplot as plt

##### Fariha Imam ####

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction, ytest), ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / np.sqrt(ytest.shape[0])


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 5

    regressionalgs = {
        'Random': algs.Regressor,
        'Mean': algs.MeanPredictor,
        'FSLinearRegression': algs.FSLinearRegression,
        'RidgeLinearRegression': algs.RidgeLinearRegression,
        # 'KernelLinearRegression': algs.KernelLinearRegression,
        'LassoRegression': algs.LassoRegression,
        # 'LinearRegression': algs.LinearRegression,
        # 'MPLinearRegression': algs.MPLinearRegression,
        'stochasticGD':algs.stochasticGD, 
        'BatchGradientDescent':algs.BatchGradientDescent,
              
              #### Bonus Questions#####
        'Momentum':algs.MomentumLinearRegression,
        'Adam':algs.ADAMLinearRegression,
       
      
        
    }
    numalgs = len(regressionalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        'FSLinearRegression': [
            { 'features': range(385)},
            { 'features': [1, 3, 5, 7, 9] },
        ],
        'RidgeLinearRegression': [
            { 'regwgt': 0.00 },
            { 'regwgt': 0.01 },
            { 'regwgt': 0.05 },
            { 'features': range(385) },
        ]
    }

    errors = {}
    y={}
    for learnername in regressionalgs:
        # get the parameters to try for this learner
        # if none specified, then default to an array of 1 parameter setting: None
        params = parameters.get(learnername, [ None ])
        errors[learnername] = np.zeros((len(params), numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0], r))

        for learnername, Learner in regressionalgs.items():
            params = parameters.get(learnername, [ None ])
            for p in range(len(params)):
                learner = Learner(params[p])
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
      #This command will the used only with stochasticGD when plotting error vs epcohs
                #y[learnername] = learner.forplot()
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p, r] = error


    for learnername in regressionalgs:
        params = parameters.get(learnername, [ None ])
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(len(params)):
            aveerror = np.mean(errors[learnername][p, :])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
        
        #standard error 
        standarderror=np.std((errors[learnername][bestparams, :]) / math.sqrt(numruns))
        # Extract best parameters
        
        best = params[bestparams]
        print ('Best parameters for ' + learnername + ': ' + str(best))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))
        print ('Standard error for ' + learnername + ': '+ str(standarderror))
    # 2(f)
    # error versus epochs for stochastic gradient descent(use when running stochastic)
    #plt.plot(np.arange(1000) , y["stochasticGD"])
    #plt.show()