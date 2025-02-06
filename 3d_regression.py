import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
    
    
    print("Data points: ", len(data['y']))

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
    ax.set_xlabel("Batch Size (x1)")
    ax.set_ylabel("KV Cache Size (x2)")
    ax.set_zlabel("Decoding Latency (y)")

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
    ax.set_xlabel("Batch Size (x1)")
    ax.set_ylabel("KV Cache Size (x2)")
    ax.set_zlabel("Decoding Latency (y)")

    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

# Main execution with new data
if __name__ == "__main__":
    # 새 데이터: 각 리스트는 제공된 표의 각 열에 해당합니다.
    memory_bound_data = {
        'x1': [
            1, 2, 4, 8, 16, 32, 48,
            1, 2, 4, 8, 16, 32, 64, 96, 192,
            1, 2, 4, 8, 16, 32, 64, 96, 192,
            1, 2, 4, 8, 16, 32, 64, 96, 192,
            1, 2, 4, 8, 16, 32, 64, 96, 192,
            1, 2, 4, 8, 16, 32, 64, 96, 192,
            1, 2, 4, 8, 16, 32, 64, 96, 192,
            1, 2, 4, 8, 16, 32, 64, 96, 192
        ],
        'x2': [
            4096, 8192, 16384, 32768, 65536, 131072, 196608,
            2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608, 393216,
            1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 2816.064,
            512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 98304,
            256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 49152,
            127, 254, 508, 1016, 2032, 4064, 8128, 12192, 24384,
            63, 126, 252, 504, 1008, 2016, 4032, 6048, 12096,
            31, 62, 124, 248, 496, 992, 1984, 2976, 5952
        ],
        'y': [
            12.667, 13.582, 14.555, 16.537, 19.016, 26.034, 33.623,
            12.17, 12.599, 13.455, 15.162, 16.384, 20.736, 28.777, 35.083, 50.875,
            12.207, 12.375, 12.895, 14.07, 15.39, 17.41, 23.206, 27.456, 43.447,
            11.755, 12.019, 12.483, 13.431, 14.407, 17.173, 21.055, 23.632, 35.834,
            11.56, 11.864, 12.408, 12.891, 13.578, 16.281, 20.213, 22.689, 33.095,
            11.57, 11.601, 12.422, 12.923, 13.108, 15.555, 19.711, 21.628, 35.023,
            11.546, 11.64, 12.386, 12.781, 13.025, 15.423, 19.202, 22.164, 33.476,
            11.485, 11.582, 12.222, 12.793, 12.976, 15.143, 19.361, 21.837, 33.337
        ]
    }
    
    # 
    compute_bound_data = {
        'x1': [
            256, 256, 256, 512, 512, 512,
            1024, 1024, 1024, 2048, 2048, 2048,
            4096, 4096, 4096
        ],
        'x2': [
            4096, 8192, 16384, 8192, 16384, 32768,
            16384, 32768, 65536, 32768, 65536, 131072,
            65536, 131072, 196608
        ],
        'y': [
            35.886, 36.099, 36.51, 63.774, 63.898, 64.371,
            121.246, 121.448, 122.149, 229.209, 229.866, 231.411,
            356.359, 357.363, 357.879
        ]
    }
    
    
    # New Data
    
    compute_bound_data_kv_cache_less_than_16 = {
    'x1': [
        256, 256, 256, 256,
        512, 512, 512, 512,
        1024, 1024, 1024, 1024,
        2048, 2048, 2048, 2048,
        4096, 4096, 4096, 4096
    ],
    'x2': [
        256, 512, 1024, 2048,
        512, 1024, 2048, 4096,
        1024, 2048, 4096, 8192,
        2048, 4096, 8192, 16384,
        4096, 8192, 16384, 32768
    ],
    'y': [
        34.107, 34.139, 34.278, 34.668,
        60.014, 60.113, 60.265, 60.947,
        113.857, 113.742, 114.973, 115.392,
        214.864, 215.671, 216.308, 218.423,
        335.265, 335.227, 337.227, 339.963
    ]
    }
    
    
    

    # 모델 실행
    best_model = try_models(compute_bound_data_kv_cache_less_than_16)

    # 결과 출력
    print(f"  Name: {best_model['name']}")
    print(f"  MSE: {best_model['mse']:.4f}")
    print(f"  R^2: {best_model['r2']:.4f}")
    print(f"  Coefficients: {best_model['coefficients']}")
    print(f"  Intercept: {best_model['intercept']}")
