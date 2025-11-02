import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy
from MLP import MLP
from public_test import compute_cost_test, predict_test


x,y = load_data('data/ex3data1.mat')
theta1, theta2 = load_weights('data/ex3weights.mat')

#TO-DO: calculate a testing a prediction and cost.
# Ejercicio 1.
# Crear la red neuronal.
twilightsparkle = MLP(theta1, theta2)
# Realizar el feedforward (Obtener las activaciones de cada capa).
a1,a2,a3,z2,z3 = twilightsparkle.feedforward(x)
p = twilightsparkle.predict(a3)
predict_test(p, y, accuracy)

# Ejercicio 2.
# One hot encoding de las etiquetas.
y_o_h = one_hot_encoding(y)
compute_cost_test(twilightsparkle, a3, y_o_h)
