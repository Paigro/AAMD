using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(float[] input)
    {
        List<float[,]> weights = mlpParameters.GetCoeff();
        List<float[]> biases = mlpParameters.GetInter();

        // Comprobación de entrada
        Debug.Log("FeedForward input length: " + input.Length);

        float[] activation = input;

        for (int layer = 0; layer < weights.Count; layer++)
        {
            float[,] W = weights[layer];
            float[] b = biases[layer];

            int outSize = W.GetLength(0);
            int inSize = W.GetLength(1);

            // Sanity check
            if (activation.Length != inSize)
            {
                Debug.LogError($"Shape mismatch at layer {layer}: activation length {activation.Length} != expected {inSize}");
                return new float[0];
            }

            float[] z = new float[outSize];

            for (int i = 0; i < outSize; i++)
            {
                float sum = b[i];
                for (int j = 0; j < inSize; j++)
                {
                    sum += W[i, j] * activation[j];
                }
                z[i] = sum;
            }

            if (layer < weights.Count - 1)
            {
                // Ocultas -> sigmoid
                float[] nextActivation = new float[outSize];
                for (int i = 0; i < outSize; i++)
                {
                    nextActivation[i] = sigmoid(z[i]);
                }
                activation = nextActivation;
            }
            else
            {
                // Capa final -> softmax
                activation = SoftMax(z);
            }
        }

        return activation;
    }

    /// <summary>
    /// Calculo de la sigmoidal
    /// </summary>
    /// <param name="z"></param>
    /// <returns></returns>
    private float sigmoid(float z)
    {
        // Prevenir overflow/underflow
        if (z < -45f) return 0f;
        if (z > 45f) return 1f;

        return 1.0f / (1.0f + Mathf.Exp(-z));
    }


    /// <summary>
    /// CAlculo de la soft max, se le pasa el vector de la ulrima capa oculta y devuelve el mismo vector, pero procesado
    /// aplicando softmax a cada uno de los elementos
    /// </summary>
    /// <param name="zArr"></param>
    /// <returns></returns>
    public float[] SoftMax(float[] zArr)
    {
        // Encontrar el valor máximo para estabilidad numérica
        float max = zArr[0];
        for (int i = 1; i < zArr.Length; i++)
        {
            if (zArr[i] > max)
            {
                max = zArr[i];
            }
        }

        // Calcular exp(z - max) y la suma
        float[] expValues = new float[zArr.Length];
        float sumExp = 0f;

        for (int i = 0; i < zArr.Length; i++)
        {
            expValues[i] = Mathf.Exp(zArr[i] - max);
            sumExp += expValues[i];
        }

        // Normalizar para obtener probabilidades
        float[] result = new float[zArr.Length];
        for (int i = 0; i < zArr.Length; i++)
        {
            result[i] = expValues[i] / sumExp;
        }

        return result;
    }

    /// <summary>
    /// Elige el output de mayor nivel
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el índice de mayor valor.
    /// </summary>
    /// <param name="output"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;
        // Buscar el índice con el valor máximo
        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}
