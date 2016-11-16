# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:39:38 2016

@author: Meng Yuxian

This is an implementation of <Improved boosting algorithms using 
confidence-rated predictions>, Schapire, 1999.
"""

from math import e, log
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Adaboost():
    """
    Adaboost(X, y, estimator = DecisionTreeClassifier, itern = 20, mode = "si
    gn")   
    Basic Adaboost to solve two-class problem
    
    Parameters
    ----------
    X: numpy 2d array (m samples * n features)
    y: numpy 1d array (m samples' label) 
    estimator: base_estimator of boosting
    itern: number of iterations
    mode: sign mode output label directly, while num mode output a confidence 
    rate x. The more positive x is ,the more likely the label is Adaboost.cls0;
    the more negative x is, the more likely the label is not Adaboost.cls0

    e.g.
    >>> x = np.array([[1,2,3,4],[2,3,4,5],[6,7,8,9],[2,5,7,8]])
    >>> y = np.array([1,2,2,1])
    >>> clf = Adaboost(x, y, mode = "num")
    >>> clf.predict(np.array([[1,7,2,8],[2,5,6,9]]))
    array([ 27.5707191 ,  32.16583895])
    >>> clf.cls0
    1
    >>> clf = Adaboost(x, y, mode = "sign")
    >>> clf.predict(np.array([[1,7,2,8],[2,5,6,9]]))
    array([ 1.,  1.])
    Note that outputs of clf.predict in num model are positive, so outputs of
    clf.predict in sign model are both clf.cls0, which is label 1.
    
    Methods
    -------
    predict
    score
    
    See also
    --------
    Adaboost

    References
    ----------
    <Improved boosting algorithms using confidence-rated predictions>, Schapire
    , 1999
    """
    def __init__(self, X, y, estimator = DecisionTreeClassifier, itern = 20, mode = "sign"):
        self.X = X
        self.y = y.copy()
        self.estimator = estimator
        self.mode = mode
        self.itern = itern
        self.estimators = [] # estimators produced by boosting algorithm
        self.alphas = np.array([])  # weights of each boost estimator
        self.m = self.X.shape[0] # number of samples
        self.w = np.array([1/self.m] * self.m) # weights of samples
        self.cls_list = [] # list used to store classes' name and numbers
        self.cls0 = y[0]
        for i in range(self.m):
            if y[i] not in self.cls_list:
                self.cls_list.append(y[i])
            if y[i] == self.cls0:
                self.y[i] = 1
            else:
                self.y[i] = -1
        if len(self.cls_list) != 2:
            raise TypeError(
            '''This Adaboost only support two-class problem, for multiclass 
            problem, please use AdaboostMH.''')
        self.train()

    def train(self):
        m = self.m
        for k in range(self.itern):
            cls = self.estimator(max_depth = 3, presort = True)
            cls.fit(self.X, self.y, sample_weight = self.w)
            self.estimators.append(cls)
            y_predict = cls.predict(self.X) 
            error = 0  # number of wrong prediction
            for i in range(m):
                if y_predict[i] != self.y[i]:
                    error += self.w[i]
            if error == 0:
                error += 0.01 # smoothness
            alpha = 0.5*log((1-error)/error) # estimator weight
            self.alphas = np.append(self.alphas, alpha)
            for i in range(m): # update sample weights
                if y_predict[i] != self.y[i]:
                    self.w[i] *= e**alpha
                else:
                    self.w[i] /= e**alpha
            self.w /= sum(self.w)

    def predict(self, X):
        y_predict = np.array([])
        if self.mode == "sign":
            for i in range(X.shape[0]):
                predict_i = (sum(self.alphas * 
                                 np.array([int(self.estimators[k].predict(X[i].reshape(1,-1))) for k in range(len(self.alphas))])))
                y_predict = np.append(y_predict, self.transfer(np.sign(predict_i)))
        else:
            for i in range(X.shape[0]):
                predict_i = (sum(self.alphas * 
                                 np.array([int(self.estimators[k].predict(X[i].reshape(1,-1))) for k in range(len(self.alphas))])))
                y_predict = np.append(y_predict, predict_i)
            
        return y_predict
    
    def transfer(self, l):
        """turn -1/+1 to previous initial label name"""
        if l == 1:
            return self.cls0
        else:
            return self.cls_list[1]
    
    def score(self, X_test, y_test):
        """return precision of trained estimator on x_test and y_test"""  
        y_predict = self.predict(X_test)
        error = 0 # error
        for i in range(X_test.shape[0]):
            if y_predict[i] != y_test[i]:
                error += 1
        error /= X_test.shape[0]
        return 1 - error
            

class AdaboostMH():
    """
    AdaboostMH(X, y, estimator = DecisionTreeClassifier, itern = 20, mode = "si
    gn")    
    Adaboost that could solve multiclass and multilabel problem.
    
    Parameters
    ----------
    X: numpy 2d array (m samples * n features)
    y: numpy 1d array (m samples' label) 
    estimator: base_estimator of boosting
    itern: number of iterations
    mode: "sign" mode will return label directly when you use predict method, 
    while "num" mode will return an array of confidence rates x which reflects 
    how likely the labels i belongs to corresbonding sample j.
    the more positive x is, the more likely the label i belongs to sample j;
    the more negative x is, the more likely the label i doesn't belong to j.

    e.g.
    >>> x = np.array([[1,2,3,4],[2,3,4,5],[6,7,8,9],[2,5,7,8]])
    >>> y = np.array([[1,2],[2],[3,1],[2,3]])
    >>> clf = AdaboostMH(x, y, mode = "num")
    >>> clf.predict(np.array([[1,7,2,8],[2,5,6,9]]))
    array([[ 3.89458577,  3.89458577,  1.14677695],
           [-1.45489964,  1.51029301,  7.75042082]])
    
    Methods
    -------
    predict
    score
    
    See also
    --------
    Adaboost

    References
    ----------
    <Improved boosting algorithms using confidence-rated predictions>, Schapire
    , 1999
    
    """
    def __init__(self, X, y, estimator = DecisionTreeClassifier, itern = 20, mode = "sign"):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.itern = itern
        self.mode = mode
        self.m = self.X.shape[0] # number of samples
        self.cls_list = [] # list used to store classes' name and numbers
#        if type(y[0]) != np.ndarray:
#           self.y = y.reshape(len(y),-1)
        for i in range(self.m):
            for cls in self.y[i]:
                if cls not in self.cls_list:
                    self.cls_list.append(cls)
        self.k = len(self.cls_list) # number of classes
        self.boost = self.train()

    def train(self):
        X = self.X
        new_X = [] #from initial problem generate new problem
        new_y = []
        for i in range(self.m):
            for cls in self.cls_list:
                new_X.append(list(X[i])+[cls])
                if cls in self.y[i]:
                    new_y.append(1)
                else:
                    new_y.append(-1)
        new_X = np.array(new_X)
        new_y = np.array(new_y)
        boost = Adaboost(new_X, new_y, estimator = self.estimator, itern = self.itern, mode = self.mode)
        return boost
    
    def predict(self, X):
        """Use trained model to predict new X
        clf.predict(x)
        """
        y_predict = []
        if self.mode == "sign":
            for i in range(X.shape[0]):
                y = []
                for cls in self.cls_list:
                    new_X = np.append(X[i], cls).reshape(1,-1)
                    predict = int(self.boost.predict(new_X))
                    if predict == 1:
                        y.append(cls)
                y_predict.append(y)
        else:
            for i in range(X.shape[0]):
                y = []
                for cls in self.cls_list:
                    new_X = np.append(X[i], cls).reshape(1,-1)
                    predict = self.boost.predict(new_X)[0]
                    y.append(predict)
                y_predict.append(y)
        y_predict = np.array(y_predict)
        return y_predict
    
    def score(self, X_test, y_test):
        """return precision of trained estimator on test dataset X and y"""  
        if self.mode != "sign":
            raise TypeError("score only support sign mode")
        y_predict = self.predict(X_test)
        error = 0 # error
        for i in range(X_test.shape[0]):
            for cls in self.cls_list:
                if cls in y_test[i]:
                    if cls not in y_predict[i]:
                        error += 1
                else:
                    if cls in y_predict[i]:
                        error += 1
        error /= (X_test.shape[0] * self.k)
        return 1 - error
        
class AdaboostMO():
    """
    AdaboostMO(X, y, code_dic = None, estimator = DecisionTreeClassifier, itern
    = 20)
    A multiclass version of Adaboost based on output codes to solve singlelabel
    problem
    
    Parameters
    ----------
    X: numpy 2d array (m samples * n features)
    y: numpy 1d array (m samples' label) 
    code_dic: dictionary (key:label, value: numpy array of -1/+1)
    estimator: base_estimator of boosting
    itern: number of iterations
    
    e.g.
    >>> x = np.array([[1,2,3,4],[2,3,4,5],[6,7,8,9],[2,5,7,8]])
    >>> y = np.array([1,2,3,1])
    >>> clf = AdaboostMO(x, y, code_dic = {1:np.array([1,-1,-1], 2:np.array([-1
    ,1,-1], 3:np.array([-1,-1,1])))}, itern = 15)
    >>> clf.predict(np.array([[1,7,2,8],[2,5,6,9]]))
    array([1,1])
    
    Methods
    -------
    predict
    score
    
    See also
    --------
    AdaboostMH

    References
    ----------
    <Improved boosting algorithms using confidence-rated predictions>, Schapire
    , 1999
    
    """
    def __init__(self, X, y, code_dic = None, estimator = DecisionTreeClassifier, itern = 20):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.itern = itern
        self.m = self.X.shape[0] # number of samples
        self.cls_list = [] # list used to store classes' name and numbers
        for i in range(self.m):
            if y[i] not in self.cls_list:
                self.cls_list.append(y[i])
        if code_dic != None:
            self.k = len(code_dic[cls_list[0]]) # dimension of encoding space
        else:
            self.k = len(self.cls_list)
            if code_dic == None: # generate default encode dictionary
                code_dic = {} 
                for i in range(self.k):
                    code = np.array([-1] * self.k)
                    code[i] = 1
                    code_dic[self.cls_list[i]] = code
        self.code_dic = code_dic #store {label: array-like code}
        self.boost = self.train()
    
    def train(self):
        y = self.encode(self.y) #encoding y and train it as AdaboostMH in num mode 
        for i in range(self.m):
            y[i] = [k for k in range(self.k) if y[i][k] == 1]
        boost = AdaboostMH(self.X, y, estimator = self.estimator, itern = self.itern, mode = "num")
        return boost
    
    def encode(self, y):
        if not isinstance(y, np.ndarray):
            return self.code_dic[y]
        return np.array([self.code_dic[i] for i in y])       
        
    def decode(self, y):
        """decode an array_like labels"""
        decode_y = []
        for i in range(len(y)):
            for cls in self.code_dic:
                if self.code_dic[cls] == i:
                    decode_y.append(cls)
                    break
        return np.array(decode_y)
        
    def predict(self, X):
        """Use trained model to predict on new X"""
        y_predict = []
        for i in range(X.shape[0]):
            confidences = self.boost.predict(X[i].reshape(1,-1))[0]
            cls_score = [sum(self.encode(cls) * confidences)for cls in self.cls_list]
            cls = self.cls_list[cls_score.index(max(cls_score))]
            y_predict.append(cls)             
        return np.array(y_predict)
    
    def score(self, x_test, y_test):
        """return precision of trained estimator on x_test and y_test"""  
        error = 0
        y_predict = self.predict(x_test)
        for i in range(len(y_test)):
            if y_predict[i] != y_test[i]:
                error += 1
        return 1 - error/len(y_test)
            
            
    
        
        
        
        
        
        
        
        
        

            
                
                
                