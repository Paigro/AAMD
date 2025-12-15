using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public struct OHE_Elements
{
    public int position;
    public int count;

    public OHE_Elements(int p, int c)
    {
        position = p;
        count = c;
    }
}

public class OneHotEncoding
{
    List<OHE_Elements> elements;
    Dictionary<int, int> extraElements;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public OneHotEncoding(List<OHE_Elements> e)
    {
        elements = e;
        extraElements = new Dictionary<int, int>();
        for (int i = 0; i < elements.Count; i++)
        {
            int pos = elements[i].position;
            int c = elements[i].count;
            extraElements.Add(pos, c);
        }
    }

    /// <summary>
    /// Realiza la trasformación del OHE a los elementos seleccionados.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public float[] Transform(float[] input)
    {
        List<float> output = new List<float>();
        for (int i = 0; i < input.Length; i++)
        {
            // Verificar si esta posicion necesita One-Hot Encoding
            if (IsOHEIndex(i))
            {
                // Esta posicion debe expandirse a One-Hot
                int numClasses = extraElements[i];  // Cuantas clases tiene esta feature
                int classValue = (int)input[i];     // Valor de la clase (ej: 0, 1, 2, 3...)

                // Crear el vector One-Hot de tamanyo numClasses
                for (int j = 0; j < numClasses; j++)
                {
                    if (j == classValue)
                    {
                        output.Add(1.0f);  // Posicion activa
                    }
                    else
                    {
                        output.Add(0.0f);  // Resto en 0
                    }
                }
            }
            else
            {
                // Esta posicion no necesita OHE, copiar el valor directamente.
                output.Add(input[i]);
            }
        }
        return output.ToArray();
    }

    /// <summary>
    /// Comprueba si necesita OneHotEncoding.
    /// </summary>
    internal bool IsOHEIndex(int i)
    {
        return (extraElements.ContainsKey(i));
    }
}
