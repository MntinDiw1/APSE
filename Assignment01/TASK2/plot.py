import matplotlib.pyplot as plt
import numpy as np
import os

filename = 'LinearDataset.csv'
output_plot = 'LinearRegressionPlot.png'

if not os.path.exists(filename):
    print(f"!!! ERROR: {filename} not found.")
else:
    # Load data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    
    x = data[:, 0]
    y = data[:, 1]
    m = data[0, 2]
    b = data[0, 3]
    m_gd = data[0, 4]
    b_gd = data[0, 5]

    # Create the line of best fit
    line_x = np.array([min(x), max(x)])
    line_y = m * line_x + b
    line_y_gd = m_gd * line_x + b_gd

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Generated Data')
    plt.plot(line_x, line_y, color='red', linewidth=4, label=f'Solution: y = {m:.2f}x + {b:.2f}')
    plt.plot(line_x, line_y_gd, color='green', linewidth=2, label=f'Gradient Descent: y = {m_gd:.2f}x + {b_gd:.2f}')
    
    plt.title("Gradient Descent Result vs normal(C++)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SAVE the plot to the same folder
    plt.savefig(output_plot, dpi=300) 
    print(f"Success: Plot saved as {output_plot}")
    
    # Optional: Keep show() if you still want the window to pop up
    # plt.show() 
    plt.close()