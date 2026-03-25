# Assignment 3
# GROUP 2: TRANSFORMATIONS


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


# PART A: RECTANGLE TRANSFORMATIONS


# 1. Original rectangle points
rectangle = np.array([
    [1, 1],
    [4, 1],
    [4, 3],
    [1, 3]
])

# Close the shape
rectangle_closed = np.vstack([rectangle, rectangle[0]])

# 2. Plot original rectangle
plt.figure(figsize=(6,6))
plt.plot(rectangle_closed[:,0], rectangle_closed[:,1], 'b-', label='Original')
plt.scatter(rectangle[:,0], rectangle[:,1], color='blue')

# 3. Scaling matrix
S = np.array([
    [2, 0],
    [0, 0.5]
])

# 4. Apply scaling
scaled = rectangle @ S.T
scaled_closed = np.vstack([scaled, scaled[0]])

# 5. Plot scaled rectangle
plt.plot(scaled_closed[:,0], scaled_closed[:,1], 'r-', label='Scaled')
plt.scatter(scaled[:,0], scaled[:,1], color='red')

# 6. Rotation matrix (45 degrees)
theta = np.radians(45)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# 7. Apply rotation
rotated = rectangle @ R.T
rotated_closed = np.vstack([rotated, rotated[0]])

# Plot rotated rectangle
plt.plot(rotated_closed[:,0], rotated_closed[:,1], 'g-', label='Rotated (45°)')
plt.scatter(rotated[:,0], rotated[:,1], color='green')

plt.title("Rectangle Transformations (Original, Scaled, Rotated)")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()



# PART B: MNIST DIGIT TRANSFORMATIONS


# 1. Load dataset
(x_train, y_train), (_, _) = mnist.load_data()

# 2. Select one digit
digit = x_train[0]

# Convert to coordinate points
points = []
for i in range(28):
    for j in range(28):
        if digit[i, j] > 0:
            points.append([j, 28 - i])  # flip y-axis

points = np.array(points)

# 3. Plot original digit
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], s=5, label='Original')

# 4. Rotate digit
rotated_digit = points @ R.T

# 5. Reflection matrix (y-axis)
M = np.array([
    [-1, 0],
    [0, 1]
])

reflected_digit = points @ M.T

# 6. Plot transformed digits
plt.scatter(rotated_digit[:,0], rotated_digit[:,1], s=5, label='Rotated')
plt.scatter(reflected_digit[:,0], reflected_digit[:,1], s=5, label='Reflected')

plt.title("MNIST Digit Transformations")
plt.legend()
plt.grid()
plt.show()



# PRINT EXPLANATION OUTPUT


print("\n=== EXPLANATION ===")
print("Scaling changes size: wider (Sx=2) and shorter (Sy=0.5).")
print("Rotation changes orientation but keeps shape size.")
print("For digits, rotation and reflection can affect recognition.")
print("Example: 6 may look like 9 when rotated.")
print("These transformations are important in machine learning and image processing.")
