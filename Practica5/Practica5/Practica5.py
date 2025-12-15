from pathlib import Path
from Utils import move_file_to_dir, one_hot_encoding, display_pca_classes,load_multiple_csv, zscore_normalize_features,display_confussion_matrix, export_normalization_params, ExportAllformatsMLPSKlearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier#, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from MLP import MLP

# Columnas a usar del dataset
columns = [
    "NEIGHBORHOOD_UP","NEIGHBORHOOD_DOWN",
    "NEIGHBORHOOD_RIGHT","NEIGHBORHOOD_LEFT",
    "NEIGHBORHOOD_DIST_UP","NEIGHBORHOOD_DIST_DOWN",
    "NEIGHBORHOOD_DIST_RIGHT","NEIGHBORHOOD_DIST_LEFT",
    #"COMMAND_CENTER_X","COMMAND_CENTER_Y",
    "AGENT_1_X","AGENT_1_Y",
    "AGENT_2_X","AGENT_2_Y",
    "CAN_FIRE",#"HEALTH",
    #"LIFE_X","LIFE_Y",
    "EXIT_X","EXIT_Y"
]
# Columnas a normalizar con zscore
label_column = "action"

####################### procesamiento de datos ###########################
def get_data(directory_path, feature_names):
    """
    Obtiene todos los datos en el directorio especificado y los normaliza.
    Args:
        directory_path (str or Path): ruta al directorio
        feature_names (list of str): nombres de las columnas a usar como features
    Returns:
        tuple: (X_norm, mu, sigma) donde X_norm es el array normalizado,
               mu es la media de cada feature, y sigma es la desviacion estandar de cada feature
    """
    # Verificamos que el directorio existe.
    dir_path = Path(directory_path)
    # Si no existe o no es un directorio, lanzamos un error.
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {directory_path}")
    # Metemos todos los archivos csv en una lista (tambien subcarpetas).
    csv_files = [f for f in dir_path.rglob('*.csv')]
    # Obtenemos los datos combinados de todos los archivos CSV.
    data = load_multiple_csv(csv_files)
    if data.shape[0] == 0:
        raise ValueError(f"No se encontraron datos en los archivos CSV del directorio: {directory_path}")
    
    # Filtramos solo las filas validas.
    # Seleccionamos solo las columnas que nos interesan.
    data = data[feature_names]
    data_values = data.values

    #FORZAMOS QUE TODOS LOS DATOS SEAN NUMERICOS.
    # Convertir a numeric, forzando errores a NaN.
    data_numeric = pd.DataFrame(data_values).apply(pd.to_numeric, errors='coerce').values
    # Eliminar filas con NaN.
    valid_rows = ~np.isnan(data_numeric).any(axis=1)    # Creamos una mascara de filas validas (sin NaN).
    data_clean = data_numeric[valid_rows]   # Filtramos solo las filas validas.
    print(f"Filas antes: {len(data_numeric)}, después: {len(data_clean)}")
    
    # Separamos input (X) y output (y)
    X = data_clean[:, :-1]  # Todas las columnas menos la ultima.
    y = data_clean[:, -1]   # Solo la ultima columna. 
    
    # Normalizamos SOLO los inputs (X)
    X_norm, mu, sigma = zscore_normalize_features(X)
    # One-hot encoding de las etiquetas (y)
    y_onehot = one_hot_encoding(y)

    # Devolvemos X normalizada, y (etiquetas originales), y los parametros de escala del zscore
    return X_norm, y_onehot, y, mu, sigma

def read_files(file_path, dest_path, overwrite=False):
    """
    Mueve todos los archivos CSV de `file_path` a `dest_path`.
    Crea `dest_path` si no existe.
    Args:
        file_path (str or Path): ruta al directorio origen
        dest_path (str or Path): ruta al directorio destino
        overwrite (bool): si True, sobrescribe si el archivo destino ya existe
    Returns:
        tuple: (X_norm, mu, sigma) donde X_norm es el array normalizado,
               mu es la media de cada feature, y sigma es la desviacion estandar de cada feature
    """
    src_dir = Path(file_path)
    if not src_dir.exists():
        raise FileNotFoundError(f"Directorio de origen no existe: {file_path}")
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() == '.csv':
            try:
                new_path, moved = move_file_to_dir(f, dest_path, overwrite=overwrite)
                if moved:
                    print(f'Movido: {f.name} -> {new_path}')
                #else:
                    #print(f'Omitido: {f.name} no se movió.')
            except Exception as e:
                print(f'No se ha podido mover {f.name}: {e}')

    # Cargamos y retornamos los datos normalizados desde el directorio destino.
    return get_data("./data", columns + [label_column])

######################## Entrenamiento y evaluacion ###########################
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Modelo"):
    '''
    Función que evalua e imprime los resultados del modelo.
    '''

    print(f"\n{'='*60}")
    print(f"Evaluando {model_name}")
    print(f"{'='*60}")
    
    # Detectar si y es one-hot o etiquetas directas
    is_onehot = len(y_train.shape) > 1 and y_train.shape[1] > 1
    
    # ==================== PREDICCIONES ====================
    # Obtener predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convertir predicciones a índices si son one-hot
    if len(y_pred_train.shape) > 1 and y_pred_train.shape[1] > 1:
        y_pred_train = y_pred_train.argmax(axis=1)
        y_pred_test = y_pred_test.argmax(axis=1)
    
    # Convertir etiquetas verdaderas a índices si son one-hot
    y_true_train = y_train.argmax(axis=1) if is_onehot else y_train
    y_true_test = y_test.argmax(axis=1) if is_onehot else y_test
    
    # Calculamos la accuracy
    train_accuracy = np.mean(y_pred_train == y_true_train)
    test_accuracy = np.mean(y_pred_test == y_true_test)
    
    print(f"Precisión (Accuracy) en ENTRENAMIENTO: {train_accuracy*100:.2f}%")
    print(f"Precisión (Accuracy) en TEST: {test_accuracy*100:.2f}%")
    
    print(f"{'='*60}\n")

    pca_filename = f"Graphics/PCA/pca_by_prediction_{model_name.lower().replace(' ', '_')}.png"
    display_pca_classes(X_test, y_pred_test, name=pca_filename, tittle=f"PCA 2D por predicciones - {model_name} - (Coloreado por Accion)")
    # Mostrar matriz de confusión
    cm_filename = f"Graphics/confussion_matrix/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    display_confussion_matrix(y_true_test, y_pred_test, cm_filename, tittle=f"Matriz de Confusión - {model_name}")

def MLP_sklearn(X_train, y_train, X_test, y_test):
    """
    Función principal para entrenar y evaluar un modelo MLP usando sklearn.
    """
    '''
    Con estos hyperparametros da un 79% aprox de accuracy.
    model = MLPClassifier(
        hidden_layer_sizes=(120, 280, 100),
        hidden_layer_sizes=(100, 240, 80),
        activation='logistic',           
        alpha=0.001,                 
        learning_rate_init=0.01,
        max_iter=3000,                
        random_state=1234
        )
    '''
    model = MLPClassifier(
        hidden_layer_sizes=(16, 100, 240, 80),
        activation='logistic',           
        alpha=0.01,                 
        learning_rate_init=0.001,
        max_iter=2000,                
        random_state=1234
        #verbose=100
    )

    print("Entrenando modelo MLP SKLearn...")
    model.fit(X_train, y_train)
    # Evaluacion del modelo.    
    evaluate_model(model, X_train, y_train, X_test, y_test, 
                            model_name="MLP SKLearn")
    return model

def MLP_custom(X_train, y_train, X_test, y_test):
    """
    Función principal para entrenar y evaluar un modelo MLP personalizado.
    """
    model = MLP(
        input_size=X_train.shape[1],
        hiden_layers=[160, 300, 140],
        output_size=y_train.shape[1],
        seed=1234
    )
    print("Entrenando modelo MLP Custom...")
    model.backpropagation(
        x=X_train, y=y_train, 
        alpha=2.5,
        lambda_=0.0001,
        numIte=2000,
        verbose=500
    )

    # Evaluacion del modelo.
    evaluate_model(model, X_train, y_train, X_test, y_test, 
                            model_name="MLP Custom")
    return model

def KNN(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(
        n_neighbors=10
        )
    print("Entrenando modelo KNN...")
    model.fit(X_train, y_train)

    # Evaluacion del modelo.
    evaluate_model(model, X_train, y_train, X_test, y_test, 
                            model_name="KNN")
    return model

def decission_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(
        max_depth=100,
        random_state=1234
    )
    print("Entrenando modelo Decission Tree...")
    model.fit(X_train, y_train)
    # Evaluacion del modelo.

    evaluate_model(model, X_train, y_train, X_test, y_test, 
                            model_name="Decission_Tree")
    return model

def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=100,
        random_state=1234
    )
    
    print("Entrenando modelo Random Forest...")
    model.fit(X_train, y_train)
    # Evaluacion del modelo.

    evaluate_model(model, X_train, y_train, X_test, y_test, 
                            model_name="random_forest")
    return model

#################### Exportacion del modelo ###########################

def export_model(model, X_train, mu, sigma):
    """
    Exporta el modelo entrenado a un archivo.
    Args:
        model: modelo entrenado
        filename (str): nombre del archivo donde guardar el modelo
    """
    # Tomamos una muestra de X para la exportacion
    X_sample = X_train[:1]

    # Exportar el Modelo a múltiples formatos (Ejercicio 5, Parte A)
    ExportAllformatsMLPSKlearn(
        model, 
        X_sample, 
        "model.pkl",           # Pickle
        "model.onnx",          # ONNX binario
        "model.json",          # ONNX JSON
        "model_custom.txt"     # Formato para Unity
    )

    # Exportar los parametros de normalizacion a JSON
    export_normalization_params("normalization_params.txt", mu, sigma)

if __name__ == "__main__":
    # Origen y destino de los archivos CSV
    org_path = "./../"
    dest_path = "./data/destination_csvs"
    # Mover los archivos CSV y obtener los datos normalizados
    X_norm, y_onehot, y, mu, sigma = read_files(file_path=org_path, dest_path=dest_path)
    # Visualizar los datos normalizados usando PCA
    display_pca_classes(X_norm, y, name="pca.png")
    print(f"\n{'='*50}")
    print(f"        |   Datos cargados y normalizados   | ")
    print(f"        v      Empieza el entrenamiento     v ")
    print(f"{'='*50}")
    # Dividir en conjunto de entrenamiento y prueba (80% - 20%)
    X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_norm, y_onehot, train_size = 0.8, random_state = 1234)
    X_train_other, X_test_other, y_train_other, y_test_other = train_test_split(X_norm, y, train_size = 0.8, random_state = 1234)
    
    #print(f"Número de muestras de entrenamiento: {X_train.shape[0]}")
    #print(f"Número de muestras de prueba: {X_test.shape[0]}")
    # Entrenamiento y evaluación del modelo MLP de sklearn
    model_mlp_sklearn = MLP_sklearn(X_train_ohe, y_train_ohe, X_test_ohe, y_test_ohe)
    # Entrenamiento y evaluación del modelo MLP personalizado
    #MLP_custom(X_train_ohe, y_train_ohe, X_test_ohe, y_test_ohe)
    # Entrenamiento y evaluación del modelo KNN
    #KNN(X_train_other, y_train_other, X_test_other, y_test_other)
    # Entrenamiento y evaluación del modelo Decision Tree
    #decission_tree(X_train_other, y_train_other, X_test_other, y_test_other)
    # Entrenamiento y evaluación del modelo Random Forest
    #random_forest(X_train_other, y_train_other, X_test_other, y_test_other)
    print(f"\n{'='*50}")
    print(f"                 ^ Termina el entrenamiento ^")
    print(f"                 |   Exportamos el modelo   |")
    print(f"{'='*50}")

    export_model(model=model_mlp_sklearn, X_train=X_train_ohe, mu=mu, sigma=sigma)

    
    