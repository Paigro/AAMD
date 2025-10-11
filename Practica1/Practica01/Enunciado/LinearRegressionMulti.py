import numpy as np
import copy
import math

from LinearRegression import LinearReg

class LinearRegMulti(LinearReg):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y, w, b, lambda_):
        super().__init__(x, y, w, b)
        self.lambda_ = lambda_
        return

    def f_w_b(self, x):
        ret = x @ self.w + self.b
        return ret

    def compute_cost(self):        
        Y_pred = self.f_w_b(self.input)
        error = (1/(np.size(self.input.shape[0])*2)) * np.sum(np.square(Y_pred - self.output))
        return error + self._regularizationL2Cost()
    
    def compute_gradient(self):
        Y_pred = self.f_w_b(self.input)

        dj_dw = ((self.input.T @ (Y_pred - self.output))) / self.input.shape[0]# Derivada parcial de w respecto a x.
        dj_db = np.sum(Y_pred - self.output) / self.input.shape[0] # Derivada parcial de b respecto a x.
        
        dj_dw += self._regularizationL2Gradient()
        return dj_dw, dj_db
    
    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    
    def _regularizationL2Cost(self):
        error = np.sum(np.square(self.w)) * (self.lambda_/ (self.input.shape[0]*2))
        return error
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    
    def _regularizationL2Gradient(self):
        regularizationL2Gradient = self.lambda_ / self.input.shape[0] * self.w
        return regularizationL2Gradient

    
def cost_test_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    dw,db = lr.compute_gradient()
    return dw,db
