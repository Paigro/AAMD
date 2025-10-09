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
        cost = super().compute_cost()
        if self.lambda_ == 0:
            return cost
        return cost + self._regularizationL2Cost()
    
    def compute_gradient(self):
        Y_pred = self.f_w_b(self.input)

        dj_dw = np.sum(((Y_pred - self.output)@self.input))/np.size(self.output) # Derivada parcial de w respecto a x.
        dj_db = np.sum(Y_pred - self.output)/np.size(self.output) # Derivada parcial de b respecto a x.
        
        if self.lambda_ == 0:
            return dj_dw, dj_db
        return dj_dw + self._regularizationL2Gradient(), dj_db
    
    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    
    def _regularizationL2Cost(self):
        error = (np.square(self.w)*self.lambda_)/(np.size(self.input)*2)
        return error
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    
    def _regularizationL2Gradient(self):
        dj_db = self.lambda_/(np.size(self.output)*self.w) 
        return dj_db

    
def cost_test_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    dw,db = lr.compute_gradient()
    return dw,db
