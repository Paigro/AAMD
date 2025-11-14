from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, load_weights,one_hot_encoding, accuracy
from public_test import checkNNGradients,MLP_test_step, SKLearn_test_step
from sklearn.model_selection import train_test_split



"""
Test 1 to be executed in Main
"""
def gradientTest():
    checkNNGradients(costNN,target_gradient,0)
    checkNNGradients(costNN,target_gradient,1)


"""
Test 2 to be executed in Main
"""
def MLP_test(X_train,y_train, X_test, y_test):
    print("We assume that: random_state of train_test_split  = 0 alpha=1, num_iterations = 2000, test_size=0.33, seed=0 and epislom = 0.12 ")
    print("Test 1 Calculando para lambda = 0")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,0,2000,0.92606,2000/10)
    print("Test 2 Calculando para lambda = 0.5")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,0.5,2000,0.92545,2000/10)
    print("Test 3 Calculando para lambda = 1")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,1,2000,0.92667,2000/10)

def sklearn_test(X_train, y_train, X_test, y_test, n_hidden_neurons = 25, num_iteration = 2000, alpha = 1):
    print("We assume that: random_state of train_test_split  = 0 alpha=1, num_iterations = 2000, test_size=0.33, seed=0 and epislom = 0.12")
    print("Test 1")
    lambda_ = 0.0
    SKLearn_test_step(X_train, y_train, X_test, y_test, lambda_, n_hidden_neurons, num_iteration, alpha, 0.92606)
    print("Test 2")
    lambda_ = 0.5
    SKLearn_test_step(X_train, y_train, X_test, y_test, lambda_, n_hidden_neurons, num_iteration, alpha, 0.92545)
    print("Test 3")
    lambda_ = 1.0
    SKLearn_test_step(X_train, y_train, X_test, y_test, lambda_, n_hidden_neurons, num_iteration, alpha, 0.92667)


def main():
    print("Main program")
    #Test 1
    gradientTest()
 
    # Ejercicio 3
    # Cargamos los datos
    X, Y = load_data('./data/ex3data1.mat')

    # Dividimos los datos en entrenamiento y test, usando un 20% para test y un 80% para entrenamiento
    # random_state se usa para evitar sesgos en la división de los datos
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=42)
    # Codificamos Y_train usando One-Hot Encoding
    y_train_encoded = one_hot_encoding(Y_train)
    
    #Test 2
    MLP_test(X_train, y_train_encoded, X_test, Y_test)

    # Ejercicio 4
    # Test 3 Probamos nuestra implementacion contra la de SKLearn
    sklearn_test(X_train, Y_train, X_test, Y_test)
    
    # Ejercicio 5 (?) Estudiare si hacerlo si me da la vida :)


main()