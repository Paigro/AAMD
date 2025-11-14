import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy, confMatrix, displayData, F1Score
from MLP import MLP
from public_test import compute_cost_test, predict_test

def main():
    x,y = load_data('data/ex3data1.mat')
    theta1, theta2 = load_weights('data/ex3weights.mat')

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

    # Ejercicio 3.
    # Calculo de la matriz de confusion y F1 Score.
    y_binario = (y == 0).astype(int) # Etiquetas binarias (1 si es la clase 0, 0 en caso contrario).
    p_binario = (p == 0).astype(int) # Predicciones binarias (1 si es la clase 0, 0 en caso contrario).
    # Calculo de la matriz de confusion.
    cm = confMatrix(y_binario, p_binario)
    # Calculo de precision, recall y F1 Score.
    tn, fp, fn, tp = cm.ravel().tolist()
    f1 = F1Score(tp, fp, fn)
    print("Matriz de confusion ([[TN, FP],[FN, TP]]):")
    print(cm)
    print(f"F1Score={f1}, TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    displayData(x)

main()