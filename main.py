from pandas import read_csv, DataFrame
import math
import matplotlib.pyplot as plt

file: DataFrame = read_csv("solidos-OD.csv", names=["x", "y"])


def calculate_coefficients(data: DataFrame, col_x_name: str, col_y_name: str) -> (float, float):
    """Calculate 'b0' and 'b1' coefficients for a simple linear regression. Accepts the 'data' as a DataFrame,
    'col_x_name' as a string corresponding to the independent variable column name, and 'col_y_name' as a string
    corresponding to the dependent variable column name"""
    data[col_x_name].astype("int")
    data[col_y_name].astype("int")
    mean_x: float = data["x"].mean()
    mean_y: float = data["y"].mean()
    values: list = [0, 0]
    for x, y in data.values:
        values[0] += (x - mean_x) * (y - mean_y)
        values[1] += math.pow((x - mean_x), 2)
    b1: float = values[0] / values[1]
    b0: float = mean_y - b1 * mean_x
    return b0, b1


def calculate_r2(y_original: DataFrame, y_predicted: DataFrame) -> float:
    """Calculate 'r2' coefficient for a simple linear regression. Accepts the 'y_original' as a single column DataFrame
    containing the original data about the dependent variable used to calculate b0 and b1 coefficients, and
    'y_predicted' as a single column DataFrame containing the predicted values."""
    sse = 0
    for i in range(0, len(y_original)):
        sse += pow((y_original[i] - y_predicted[i]), 2)
    sst = 0
    mean_y = y_original.mean()
    for j in range(0, len(y_original)):
        sst += pow((y_original[j] - mean_y), 2)
    result: float = 1 - sse / sst
    return result


def calculate_rmse(y_original: DataFrame, y_predicted: DataFrame):
    """Calculate the 'rmse' coefficient. Accepts the 'y_original' as a single column DataFrame containing the original
     data about the dependent variable used to calculate b0 and b1 coefficients, and 'y_predicted' as a single column
     DataFrame containing the predicted values."""
    vals = 0
    n = len(y_original)
    for i in range(0, n):
        vals += pow(y_original[i] - y_predicted[i], 2)
    result = math.sqrt(vals / n)
    return result


coefficients: tuple = calculate_coefficients(data=file, col_x_name="x", col_y_name="y")
print(f"B0 = {coefficients[0]}, B1 = {coefficients[1]}")

y_predict: DataFrame = coefficients[0] + coefficients[1]*file["x"]
r2 = calculate_r2(y_original=file["y"], y_predicted=y_predict)
rmse = calculate_rmse(y_original=file["y"], y_predicted=y_predict)
print(f"R2: {r2}")
print(f"RMSE: {rmse}")

x_out_value: DataFrame = DataFrame([53, 58, 67])
y_out_predict: DataFrame = coefficients[0] + coefficients[1]*x_out_value

plt.figure(figsize=(14, 8))
plt.title("Simple Linear Regression", fontsize=18)
plt.grid(True)
plt.xlabel("Independent variable", fontsize=14)
plt.ylabel("Dependent variable", fontsize=14)
plt.scatter(file["x"], file["y"], color="green")
plt.scatter(x_out_value, y_out_predict, color="red")
plt.plot(file["x"], y_predict, color="blue", linewidth=2)
plt.show()
