import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def perform_multivariable_regression(data):
    """
    Perform multivariable regression to find coefficients for the equation:
    y = alpha * x1 + beta * x2 + gamma * (x1 * x2) + delta

    Parameters:
        data (dict): Dictionary with keys 'batch_size', 'kv_cache_size', and 'decode_latency_per_iteration', each containing a list of values.

    Returns:
        dict: Regression coefficients and intercept.
        LinearRegression: Fitted regression model.
    """
    # Extract data
    batch_size = np.array(data['batch_size'])
    kv_cache_size = np.array(data['kv_cache_size'])
    decode_latency_per_iteration = np.array(data['decode_latency_per_iteration'])

    # Prepare features
    X = np.column_stack([batch_size, kv_cache_size, batch_size * kv_cache_size])

    # Perform regression
    model = LinearRegression()
    model.fit(X, decode_latency_per_iteration)

    # Retrieve coefficients and intercept
    coefficients = {
        'alpha': model.coef_[0],
        'beta': model.coef_[1],
        'gamma': model.coef_[2],
        'delta': model.intercept_
    }

    return coefficients, model

def plot_regression_results(data, model, coefficients, filename="regression_plot_with_metrics.png"):
    """
    Plot the actual data points, predicted regression surface, and error metrics.

    Parameters:
        data (dict): Original data containing 'batch_size', 'kv_cache_size', and 'decode_latency_per_iteration'.
        model (LinearRegression): Fitted regression model.
        coefficients (dict): Regression coefficients.
        filename (str): Name of the file to save the plot.
    """
    batch_size = np.array(data['batch_size'])
    kv_cache_size = np.array(data['kv_cache_size'])
    decode_latency_per_iteration_actual = np.array(data['decode_latency_per_iteration'])

    # Prepare grid for surface
    batch_size_range = np.linspace(min(batch_size), max(batch_size), 100)
    kv_cache_size_range = np.linspace(min(kv_cache_size), max(kv_cache_size), 100)
    batch_size_grid, kv_cache_size_grid = np.meshgrid(batch_size_range, kv_cache_size_range)
    batch_kv_grid = batch_size_grid * kv_cache_size_grid

    # Predict using the regression model
    X_grid = np.column_stack([batch_size_grid.ravel(), kv_cache_size_grid.ravel(), batch_kv_grid.ravel()])
    decode_latency_per_iteration_pred_grid = model.predict(X_grid).reshape(batch_size_grid.shape)

    # Predict on actual data
    X_actual = np.column_stack([batch_size, kv_cache_size, batch_size * kv_cache_size])
    decode_latency_per_iteration_pred_actual = model.predict(X_actual)

    # Calculate metrics
    mse = mean_squared_error(decode_latency_per_iteration_actual, decode_latency_per_iteration_pred_actual)
    r2 = r2_score(decode_latency_per_iteration_actual, decode_latency_per_iteration_pred_actual)

    # Plot actual data points
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(batch_size, kv_cache_size, decode_latency_per_iteration_actual, color='blue', label='Actual Data')

    # Plot regression surface
    ax.plot_surface(batch_size_grid, kv_cache_size_grid, decode_latency_per_iteration_pred_grid, color='red', alpha=0.5, label='Regression Surface')

    # Labels and title
    ax.set_title('Multivariable Regression: Decoding Latency')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('KV Cache Size')
    ax.set_zlabel('Decoding Latency')

    # Add metrics text
    textstr = f"MSE: {mse:.4f}\nR^2: {r2:.4f}"
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save plot to file
    plt.savefig(filename)
    print(f"Plot with metrics saved as {filename}")

    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'batch_size': [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4],
        'kv_cache_size': [513, 513, 513, 513, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 4097, 4097, 4097],
        'decode_latency_per_iteration': [16.715, 17.296, 17.467, 20.262, 17.285, 17.819, 18.673, 22.627,
                             18.024, 19.055, 21.078, 27.533, 19.304, 21.496, 27.519]
    }

    # Perform regression
    coefficients, model = perform_multivariable_regression(data)

    # Print results
    print("Regression Coefficients:")
    for key, value in coefficients.items():
        print(f"  {key}: {value:.6f}")

    # Plot results with metrics and save to file
    plot_regression_results(data, model, coefficients, filename="decode_latency_per_iteration_regression_with_metrics.png")