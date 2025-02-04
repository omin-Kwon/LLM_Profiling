import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

def try_models(data):
    """
    Test multiple regression models (linear, interaction, quadratic) to find the simplest model that fits the data.

    Parameters:
        data (dict): Dictionary with keys 'x1', 'x2', and 'y', each containing a list of values.

    Returns:
        dict: Best model details including coefficients, intercept, MSE, and R^2.
    """
    x1 = np.array(data['x1'])
    x2 = np.array(data['x2'])
    y = np.array(data['y'])

    # Prepare features for linear regression
    X_linear = np.column_stack([x1, x2])

    # Train the linear model
    model = LinearRegression()
    model.fit(X_linear, y)
    y_pred = model.predict(X_linear)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Linear Regression Model:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R^2: {r2:.4f}")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_}\n")

    # Plot the model results
    plot_model_with_plane("Linear", x1, x2, y, y_pred, mse, r2, "Linear_regression_with_plane.png", model)
    plot_model_without_plane("Linear", x1, x2, y, "Linear_regression_without_plane.png")

    return {
        'name': "Linear",
        'model': model,
        'mse': mse,
        'r2': r2,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }

def plot_model_with_plane(model_name, x1, x2, y_actual, y_pred, mse, r2, filename, model):
    """
    Plot the actual data points and the regression plane for the linear model.

    Parameters:
        model_name (str): Name of the model.
        x1 (np.ndarray): Feature x1.
        x2 (np.ndarray): Feature x2.
        y_actual (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.
        mse (float): Mean Squared Error of the model.
        r2 (float): R-squared value of the model.
        filename (str): File name to save the plot.
        model (LinearRegression): Trained linear regression model.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for actual data points
    ax.scatter(x1, x2, y_actual, color='blue', label='Actual Data')

    # Generate meshgrid for the surface plot
    x1_range = np.linspace(min(x1), max(x1), 50)
    x2_range = np.linspace(min(x2), max(x2), 50)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    X_pred_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

    # Predict values on the grid
    y_pred_grid = model.predict(X_pred_grid).reshape(x1_grid.shape)

    # Surface plot for predicted values
    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='red', alpha=0.5)

    # Labels and title
    ax.set_title("Multivariable Regression: Decoding Latency")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("KV Cache Size")
    ax.set_zlabel("Decoding Latency")

    # Add metrics text box
    textstr = f"MSE: {mse:.4f}\nR^2: {r2:.4f}"
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_model_without_plane(model_name, x1, x2, y_actual, filename):
    """
    Plot only the actual data points without the regression plane.

    Parameters:
        model_name (str): Name of the model.
        x1 (np.ndarray): Feature x1.
        x2 (np.ndarray): Feature x2.
        y_actual (np.ndarray): Actual target values.
        filename (str): File name to save the plot.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for actual data points
    ax.scatter(x1, x2, y_actual, color='blue', label='Actual Data')

    # Labels and title
    ax.set_title("Multivariable Regression: Decoding Latency (Data Points Only)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("KV Cache Size")
    ax.set_zlabel("Decoding Latency")

    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

# Example usage
if __name__ == "__main__":
    # Extended data with additional samples
    data = {
        'x1': [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4],
        'x2': [513, 1026, 2052, 4104, 1025, 2050, 4100, 8200, 2049, 4098, 8196, 16392, 4097, 8194, 16388],
        'y': [16.715, 17.296, 17.467, 20.262, 17.285, 17.819, 18.673, 22.627,
              18.024, 19.055, 21.078, 27.533, 19.304, 21.496, 27.519]
    }

    # Try models
    best_model = try_models(data)

    # Print best model details
    print("Best Model:")
    print(f"  Name: {best_model['name']}")
    print(f"  MSE: {best_model['mse']:.4f}")
    print(f"  R^2: {best_model['r2']:.4f}")
    print(f"  Coefficients: {best_model['coefficients']}")
    print(f"  Intercept: {best_model['intercept']}")