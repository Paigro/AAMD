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
        # parametros del modelo
        self.input = x  # input de entrenamiento
        self.output = y # output de entrenamiento
        self.w = w
        self.b = b
        self.m = x.shape[0]  # numero de ejemplos de entrenamiento
    """
    Realiza la prediccion de la funcion lineal con los parametros w y b para un input x.
    """
    def f_w_b(self, x):
        return np.multiply(self.w, x) + self.b
    """
    Funcion de coste para regresion lineal.
    Se calcula el error cuadratico medio entre las predicciones y los valores reales.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # compute MSE
    def compute_cost(self):
        Y_pred = self.f_w_b(self.input)
        # MSE = (pred - real)^2 / (2*m) donde m es el numero de ejemplos de entrenamiento
        error = np.sum(np.square(Y_pred - self.output))/(self.m*2)
        return error
    
    """
    Calcula la pendiente de la funcion de coste para minimizarla
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    def compute_gradient(self):
        Y_pred = self.f_w_b(self.input)

        # Derivada parcial de j respecto a w.
        dj_dw = (self.input.T @ (Y_pred - self.output)) / self.m
        # Derivada parcial de j respecto a b.
        dj_db = np.sum(Y_pred - self.output) / self.m 
        
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
        # Array de historial de coste para cada iteracion
        J_history = []
        w_history = []
        b_history = []
        w_initial = copy.deepcopy(self.w)  # avoid modifying global w within function
        b_initial = copy.deepcopy(self.b)  # avoid modifying global b within function
        # Gradient descent iteration by m examples.
        w_history.append(w_initial)
        J_history.append(self.compute_cost())
        b_history.append(b_initial)
        
        for i  in range(num_iters):
            # Pendiente de la funcion de coste respecto a w y b
            grad_w, grad_b = self.compute_gradient()
            # Modifica los parametros w y b en la direccion de la pendiente negativa
            self.w = self.w - alpha * grad_w
            self.b = self.b - alpha * grad_b

            J_history.append(self.compute_cost())
            w_history.append(self.w)
            b_history.append(self.b)

        return self.w, self.b, J_history, w_initial, b_initial


def cost_test_obj(x, y, w_init, b_init):
    lr = LinearReg(x, y, w_init, b_init)
    cost = lr.compute_cost()
    return cost

def compute_gradient_obj(x, y, w_init, b_init):
    lr = LinearReg(x, y, w_init, b_init)
    dw,db = lr.compute_gradient()
    return dw,db
