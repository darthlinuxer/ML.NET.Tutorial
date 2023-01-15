using Microsoft.ML.Data;

namespace Example1;

public class Prediction
{
    [ColumnName("Score")]
    public float PredictedValue { get; set; }
}