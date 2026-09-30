import numpy as np
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
        # y = w1*x1 + w2*x2 + ... + wn*xn + b
        return x @ self.w + self.b
    def compute_gradient(self):
        dj_dw, dj_db = super().compute_gradient()
        # Sumamos la regularizacion a la derivada parcial de w
        dj_dw += self._regularizationL2Gradient()
        return dj_dw, dj_db
    """
    Compute the regularization cost
    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    def _regularizationL2Cost(self):
        # L2 = (lambda / (2 * m)) * sum(w^2)
        return (self.lambda_ / (2 * self.m)) * np.sum(np.square(self.w))
    """
    Compute the regularization gradient
    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """     
    def _regularizationL2Gradient(self):
        # L2 = (lambda / m) * w
        return (self.lambda_ * self.w )/ self.m 

    
def cost_test_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x, y, w_init, b_init):
    lr = LinearRegMulti(x, y, w_init, b_init, 0)
    dw,db = lr.compute_gradient()
    return dw,db
