import matplotlib.pyplot as plt
import numpy as np
import os

# Use the EXACT name from your screenshot
filename = 'LinearDataset.csv'

if not os.path.exists(filename):
    print(f"!!! ERROR: {filename} not found.")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Files actually here: {os.listdir('.')}")
else:
    # Load data - Column 0:x, 1:y, 2:slope, 3:intercept
    # We skip the header row
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    
    x = data[:, 0]
    y = data[:, 1]
    m = data[0, 2]
    b = data[0, 3]

    # Create the line of best fit
    line_x = np.array([min(x), max(x)])
    line_y = m * line_x + b

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Generated Data')
    plt.plot(line_x, line_y, color='red', label=f'Solution: y = {m:.2f}x + {b:.2f}')
    
    plt.title("C++ Normal Equation Result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()