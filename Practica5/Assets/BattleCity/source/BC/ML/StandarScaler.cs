using UnityEngine;

public class StandarScaler
{
    private float[] mean;
    private float[] std;
    public StandarScaler(string serieliced)
    {
        string[] lines = serieliced.Split("\n");
        string[] meanStr = lines[0].Split(",");
        string[] stdStr = lines[1].Split(",");
        mean = new float[meanStr.Length];
        std = new float[stdStr.Length];
        for (int i = 0; i < meanStr.Length; i++)
        {
            mean[i] = float.Parse(meanStr[i], System.Globalization.CultureInfo.InvariantCulture);
        }

        for (int i = 0; i < stdStr.Length; i++)
        {
            std[i] = float.Parse(stdStr[i], System.Globalization.CultureInfo.InvariantCulture);
            std[i] = Mathf.Sqrt(std[i]);
        }
    }

    /// <summary>
    /// Aplica una normalización - media entre desviación tipica.
    /// </summary>
    /// <param name="a_input"></param>
    /// <returns></returns>
    public float[] Transform(float[] a_input)
    {
        if (a_input.Length != mean.Length || a_input.Length != std.Length)
        {
            Debug.LogError($"Input size mismatch! Input: {a_input.Length}, Expected: {mean.Length}");
            Debug.LogError($"Mean length: {mean.Length}, Std length: {std.Length}");
            return a_input; // or throw exception
        }

        float[] scaled = new float[a_input.Length];

        // Normalizamos cada elemento usando la formula de normalizacion z-score.
        for (int i = 0; i < scaled.Length; i++)
            // Z = (X - μ) / σ
            scaled[i] = (a_input[i] - mean[i])/std[i];

        return scaled;
    }
}