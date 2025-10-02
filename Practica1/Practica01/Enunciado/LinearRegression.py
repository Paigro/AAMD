import numpy as np
import copy
import math

class LinearReg:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
    """
    def __init__(self, x, y, w, b):
        #(scalar): Parameters of the model
        self.input = x
        self.output = y
        self.w = w
        self.b = b
        return #delete this return

    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    def f_w_b(self, x):
        return np.multiply(self.w, x) + self.b


    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # compute MSE
    def compute_cost(self):
        Y_pred = self.f_w_b(self.output)
        error = np.mean((self.output - Y_pred) ** 2)/2
        #error = np.multiply((1/2*len(self.output)), np.sum(np.square(self.output - Y_pred)))
        return error
    

    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    def compute_gradient(self):
        dj_dw = np.gradient(self.input, self.w) # Derivada parcial de w respecto a x.
        dj_db = np.gradient(self.input, self.b) # Derivada parcial de b respecto a x.
        return dj_dw, dj_db


    
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
        #TODO: gradient descent iteration by m examples.
        w_history[0] = w_initial
        J_history[0] = self.compute_cost()
        
        for i  in range(1, num_iters):
            pendiente_w = np.sum((self.f_w_b(self.input) - self.output) ** 2) / len(self.input)
            pendiente_b = np.sum(np.multiply(self.f_w_b(self.input) - self.output), np.multiply(2, self.input)) / len(self.input)

            self.w = w_history[i-1] - alpha * pendiente_w
            self.b = self.b - alpha * pendiente_b

            J_history[i] = self.compute_cost()
            w_history[i] = self.w


        return self.w, self.b, J_history, w_initial, b_initial


def cost_test_obj(x, y, w_init, b_init):
    lr = LinearReg(x, y, w_init, b_init)
    cost = lr.compute_cost()
    return cost

def compute_gradient_obj(x, y, w_init, b_init):
    lr = LinearReg(x, y, w_init, b_init)
    dw,db = lr.compute_gradient()
    return dw,db
