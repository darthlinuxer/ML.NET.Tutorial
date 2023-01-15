using Example1;
using Microsoft.ML;

MLContext mlContext = new();

// 1. Import or create training data
DataSource[] data = {
               new DataSource() { X = 1, Y = 1, Z = 1},
               new DataSource() { X = 2, Y = 2, Z = 2 },
               new DataSource() { X = 3, Y = 3, Z = 3 },
               new DataSource() { X = 4, Y = 4, Z = 4 }, 
               new DataSource() { X = 5, Y = 5, Z = 5 }, 
               new DataSource() { X = 6, Y = 6, Z = 6 }, 
               new DataSource() { X = 7, Y = 7, Z = 7 }, 
               new DataSource() { X = 8, Y = 8, Z = 8 },
               new DataSource() { X = 9, Y = 9, Z = 9 } 
               };

IDataView trainingData = mlContext.Data.LoadFromEnumerable(data);

// 2. Specify data preparation and model training pipeline
var pipeline1 = mlContext.Transforms.Concatenate("Features", new[] { "X", "Y" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Z",
                                               maximumNumberOfIterations: 200));

var pipeline2 = mlContext.Transforms.Concatenate("Features", new[] { "X", "Y" })
    .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Z"));

var pipeline3 = mlContext.Transforms.Concatenate("Features", new[] { "X", "Y" })
    .Append(mlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Z"));

// 3. Train model
var model1 = pipeline1.Fit(trainingData);
var model2 = pipeline2.Fit(trainingData);
var model3 = pipeline3.Fit(trainingData);

// 4. Make a prediction
var newValue = new DataSource() { X = 10, Y = 10 };
var result1 = mlContext.Model.CreatePredictionEngine<DataSource, Prediction>(model1).Predict(newValue);var result2 = mlContext.Model.CreatePredictionEngine<DataSource, Prediction>(model2).Predict(newValue);var result3 = mlContext.Model.CreatePredictionEngine<DataSource, Prediction>(model3).Predict(newValue);

Console.WriteLine($"Predicted result is {result1.PredictedValue} - algorithm SDCA");
Console.WriteLine($"Predicted result is {result2.PredictedValue} - algorithm Poisson");
Console.WriteLine($"Predicted result is {result3.PredictedValue} - algorithm Gradient");