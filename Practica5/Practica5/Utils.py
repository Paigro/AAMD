from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# Graficos
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Mover archivos
from pathlib import Path
import shutil

def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        name = parameter["name"]
        print("Capa ",name)
        if name != "classes" and name != "shape_tensor":
            print("procesando ",name)
            s += "parameter:"+str(parameterIndex)+"\n"
            print(parameter["dims"])
            s += "dims:"+str(parameter["dims"])+"\n"
            print(parameter["name"])
            s += "name:"+str(parameter["name"])+"\n"
            print(parameter["doubleData"])
            s += "values:"+str(parameter["doubleData"])+"\n"
            index = index + 1
            parameterIndex = index // 2
        else:
            print("Esta capa no es interesante ",name)
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)

def export_normalization_params(file, mean, var):
    """Exporta mean y std en el mismo formato que C# espera"""
    line = ""
    for i in range(len(mean)-1):
        line = line + str(mean[i]) + ","
    line = line + str(mean[len(mean)-1]) + "\n"
    
    for i in range(len(var)-1):
        line = line + str(var[i]) + ","
    line = line + str(var[len(var)-1]) + "\n"
    
    with open(file, 'w') as f:
        f.write(line)
    print(f"Parámetros de normalización guardados en {file}")

############### Representacion grafica de datos ####################
def display_pca_classes(X, y, name="pca.png", tittle="PCA 2D por predicciones (Coloreado por Accion)", figsize=(10, 8)):
    """
    Muestra los datos en 2D usando PCA. Los puntos se colorean segun sus etiquetas.
    """
    # Crear directorio si no existe
    output_path = Path(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reducimos a a 2 PC usando PCA
    pca = PCA(n_components=10)  # Primero ajustamos con muchas componentes para capturar varianza.
    # Reduce X a 2 dimensiones usando el pca ajustado.
    X_reduced = pca.fit_transform(X) # Además de ajustar, transforma X.
   
    # Mostrar varianza por cada componente
    #print("Varianza acumulada:", np.cumsum(pca.fit(X).explained_variance_ratio_))
    #print("Varianza total (PC1 + PC2):", pca.explained_variance_ratio_.sum())

    # Creamos la figura
    plt.figure(figsize=figsize) 

    # Obtenemos las clases.
    clases = np.unique(y)
    #print(f"Clases encontradas: {clases}")
    # Se dibujan los puntos de cada clase con un color diferente.
    for clase in clases:
        # Filtramos los puntos que pertenecen a esta clase.
        mask = (y == clase)
        #print(f"Mostrando clase {clase} con {np.sum(mask)} puntos.")
        plt.scatter(
            X_reduced[mask, 0],         # Coordenada en PC1
            X_reduced[mask, 1],         # Coordenada en PC2
            label=f'{str(clase)}',# Etiqueta para la leyenda.
            edgecolor='k',              # Color del borde de los puntos k=negro.
            s=50,                       # Tamanyo de los puntos.
            alpha=0.7                   # Transparencia para ver superposiciones.
        )
    
    plt.title(tittle)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    
    # Mostramos leyenda con todas las clases.
    plt.legend(title='Accion', bbox_to_anchor=(1, 1), loc='best')
    plt.grid() # Mostramos cuadricula porque queda bonito.
    plt.savefig(name) # Guardamos la imagen.
    plt.show() # Esto muestra la figura en una ventana para poder inspeccionarla.
    #plt.close() # Cierra la figura.

def display_confussion_matrix(y_true, y_pred, name="confussion_matrix.png", tittle="Matriz de Confusión"):
    # Crear directorio si no existe
    output_path = Path(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(tittle)
    # Mostrar la gráfica
    plt.savefig(name) # Guardamos la imagen.
    plt.show()
    #plt.close() # Cierra la figura.

############### Carga y limpieza de datos archivos CSV ####################
def load_multiple_csv(filePathList, output_file="./data/Training.csv"):
    """
    carga multiples archivos CSV y los combina en un solo array de numpy
    Args:
        filePathList (list of str): lista de rutas de archivos CSV
    Returns:
        numpy.ndarray: datos combinados de todos los archivos CSV
    """
    # Leer los archivos CSV. 
    # Con on_bad_lines='skip' ignoramos lineas con errores y evitamos errores para poder seguir trabajando con el resto de datos.
    dataframes = [pd.read_csv(filePath, on_bad_lines='skip') for filePath in filePathList]
    # Concatena verticalmente los DataFrames en uno solo.
    if len(dataframes) == 0:
        return np.array([])  # Retorna un array vacío si no hay dataframes
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    # Crear directorio de salida si no existe
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Guardar el archivo combinado
    combined_dataframe.to_csv(output_file, index=False)
    # Devolver los valores como un array de numpy.
    return combined_dataframe

def zscore_normalize_features(X):
    """
    calcula X, zcore normalizado por columna
    Args:
      X (ndarray (m,n))     : datos de entrada, m ejemplos, n features
    Returns:
      X_norm (ndarray (m,n)): entrada normalizada por columna
    """
    # Media de cada columna/feature (Para centrar datos).
    mu = np.mean(X, axis=0)
    # Desviacion estandar de cada columna/feature (Para escalar).
    sigma = np.std(X, axis=0, ddof=0)
    # Evitar division por cero en caso de desviacion estandar cero (toda la columna tiene el mismo valor).
    # Sustituimos sigma por 1 en esas posiciones (valor-valor)/1 = 0, que es correcto.
    sigma = np.where(sigma == 0, 1, sigma)
    # Input normalizado por columna.
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def one_hot_encoding(Y):
    """
    Implementacion del one hot encoding usando OneHotEncoder de sklearn.
    Args:
        Y (ndarray (m,)): vector de etiquetas de clase  
    Returns:
        YEnc (ndarray (m, num_classes)): matriz one-hot codificada
    """
    # Modificamos la forma de Y para que sea compatible con OneHotEncoder.
    Y = Y.reshape(-1, 1) 
    # Creamos el codificador.
    encoder = OneHotEncoder(sparse_output=False) 
    YEnc = encoder.fit_transform(Y)
    return YEnc


def move_file_to_dir(src, dest_dir, search_text="in", overwrite=False):
    """
    Mueve un archivo de `src` a `dest_dir`. Crea `dest_dir` si no existe.

    Args:
        src (str or Path): ruta al archivo origen
        dest_dir (str or Path): ruta al directorio destino
        overwrite (bool): si True, sobrescribe si el archivo destino ya existe

    Returns:
        str: ruta absoluta al archivo movido en el destino

    Raises:
        FileNotFoundError: si `src` no existe
        FileExistsError: si ya existe el archivo en `dest_dir` y overwrite=False
    """
    src_path = Path(src)
    # Verificamos que el archivo origen existe y es un archivo CSV.
    if not src_path.exists() or not src_path.is_file() or src_path.suffix.lower() != '.csv':
        raise FileNotFoundError(f"Archivo de origen no encontrado: {src}")

    dest_dir_path = Path(dest_dir)
    # Crear el directorio destino si no existe
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir_path / src_path.name
    try:
        # Leer el contenido del archivo
        with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Verificar si contiene el texto buscado
        if search_text not in content:
            return str(dest_path), False  # No mover el archivo si se encuentra el texto
        if dest_path.exists() and not overwrite:
            raise FileExistsError(f"El archivo destino ya existe: {dest_path}")
    except Exception as e:
        print(f"Error con {src_path.name}: {e}")

    # Mover el archivo
    shutil.move(str(src_path), str(dest_path))
    return str(dest_path), True  # Archivo movido exitosamente