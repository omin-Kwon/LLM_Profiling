import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

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

    # Prepare features for different models
    X_linear = np.column_stack([x1, x2])
    X_interaction = np.column_stack([x1, x2, x1 * x2])

    # Quadratic features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_quadratic = poly.fit_transform(X_linear)

    # Models to test
    models = {
        'Linear': X_linear,
        'Interaction': X_interaction,
        'Quadratic': X_quadratic
    }

    best_model = None
    best_r2 = -np.inf

    for name, X in models.items():
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Model: {name}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R^2: {r2:.4f}")
        print(f"  Coefficients: {model.coef_}")
        print(f"  Intercept: {model.intercept_}\n")

        # Track the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = {
                'name': name,
                'model': model,
                'mse': mse,
                'r2': r2,
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }

        # Plot the model results
        plot_model_results(name, x1, x2, y, y_pred, mse, r2, f"{name}_regression_plot.png")

    return best_model

def plot_model_results(model_name, x1, x2, y_actual, y_pred, mse, r2, filename):
    """
    Plot the actual vs predicted results for a given model and include metrics.

    Parameters:
        model_name (str): Name of the model.
        x1 (np.ndarray): Feature x1.
        x2 (np.ndarray): Feature x2.
        y_actual (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.
        mse (float): Mean Squared Error of the model.
        r2 (float): R-squared value of the model.
        filename (str): File name to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_actual)), y_actual, color='blue', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.6)

    plt.title(f"{model_name} Model: Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("y")

    # Add metrics to the plot
    textstr = f"MSE: {mse:.4f}\nR^2: {r2:.4f}"
    plt.gcf().text(0.15, 0.85, textstr, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
