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
    
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
      w_initial : (ndarray): Shape (1,) initial w value before running gradient descent
      b_initial : (scalar) initial b value before running gradient descent
    """
    def gradient_descent(self, alpha, num_iters):
        # An array to store cost J and w's at each iteration — primarily for graphing later
        J_history = []
        w_history = []
        w_initial = copy.deepcopy(self.w)  # avoid modifying global w within function
        b_initial = copy.deepcopy(self.b)  # avoid modifying global w within function
        # Gadient descent iteration by m examples.
        for i  in range(num_iters):
            new_w, new_b = self.compute_gradient()

            self.w = self.w - alpha * new_w
            self.b = self.b - alpha * new_b

            J_history.append(self.compute_cost())
            w_history.append(self.w)
        # Triquiñuela ???
        #a = self.w[1]
        #self.w[1] = self.w[2]
        #self.w[2] = a
        return self.w, self.b, J_history, w_initial, b_initial



    
def cost_test_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    dw,db = lr.compute_gradient()
    return dw,db
