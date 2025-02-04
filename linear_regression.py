import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x, y):
    """
    Perform linear regression to find coefficients a and b in y = ax + b.

    Parameters:
        x (list or np.ndarray): Array of x values.
        y (list or np.ndarray): Array of y values.

    Returns:
        tuple: Coefficients (a, b)
    """
    x = np.array(x)
    y = np.array(y)

    # Number of data points
    N = len(x)

    # Calculate sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)

    # Calculate a and b
    a = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x**2)
    b = (sum_y - a * sum_x) / N

    return a, b
  
def plot_regression(x, y, a, b, filename= "regression_plot.png"):
    """
    Plot the original data points and the linear regression line.

    Parameters:
        x (list or np.ndarray): Array of x values.
        y (list or np.ndarray): Array of y values.
        a (float): Slope of the regression line.
        b (float): Intercept of the regression line.
    """
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(x, y, color='blue', label='Data Points')

    # Plot regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Regression Line: y = {a:.6f}x + {b:.6f}')

    # Add labels and legend
    plt.title('Linear Regression Visualization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Save plot to file
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    # Show plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample data
    
    #kv cache : 512, batch = 1,2,4,8
    x_1 = [1,	2,	4,	8]
    y_1 = [16.715, 17.296, 17.467, 20.262]

     # Perform linear regression
    a_1, b_1 = linear_regression(x_1, y_1)
    # Print results
    print(f"The linear regression equation is: y = {a_1:.6f}x + {b_1:.6f}")
    plot_regression(x_1, y_1, a_1, b_1, filename="linear_regression_plot_1.png")
    
    
    
    x_2 = [1,	2,	4,	8]
    y_2 = [17.285, 17.819, 18.673, 22.627]
    # Perform linear regression
    a_2, b_2 = linear_regression(x_2, y_2)
    # Print results
    print(f"The linear regression equation is: y = {a_2:.6f}x + {b_2:.6f}")
    plot_regression(x_2, y_2, a_2, b_2, filename="linear_regression_plot_2.png")
    
    x_3 = [1,	2,	4,	8]
    y_3 = [18.024, 19.055, 21.078, 27.533]
    # Perform linear regression
    a_3, b_3 = linear_regression(x_3, y_3)
    # Print results
    print(f"The linear regression equation is: y = {a_3:.6f}x + {b_3:.6f}")
    plot_regression(x_3, y_3, a_3, b_3, filename="linear_regression_plot_3.png")
    
    x_4 = [1,	2,	4]
    y_4 = [19.304, 21.496, 27.519]
     # Perform linear regression
    a_4, b_4 = linear_regression(x_4, y_4)
    # Print results
    print(f"The linear regression equation is: y = {a_4:.6f}x + {b_4:.6f}")
    plot_regression(x_4, y_4, a_4, b_4, filename="linear_regression_plot_4.png")
    



