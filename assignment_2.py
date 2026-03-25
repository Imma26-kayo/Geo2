# GROUP 2, 
# Assignment 2
# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 


# PART A: RECTANGLE TRANSFORMATIONS

print("=== PART A: RECTANGLE TRANSFORMATIONS ===")

# Define rectangle
rectangle = np.array([
    [1,1], [4,1], [4,3], [1,3], [1,1]
])

# Scaling

S = np.array([[2, 0],
              [0, 0.5]])

scaled = rectangle[:-1] @ S.T
scaled = np.vstack([scaled, scaled[0]])

# Rotation (45 degrees)

theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

rotated = rectangle[:-1] @ R.T
rotated = np.vstack([rotated, rotated[0]])

# Plot Rectangle Transformations

plt.figure(figsize=(12,4))

# Original vs Scaled
plt.subplot(1,3,1)
plt.plot(rectangle[:,0], rectangle[:,1], 'bo-', label='Original')
plt.plot(scaled[:,0], scaled[:,1], 'ro-', label='Scaled')
plt.title("Scaling")
plt.legend()
plt.axis('equal')
plt.grid()

# Original vs Rotated
plt.subplot(1,3,2)
plt.plot(rectangle[:,0], rectangle[:,1], 'bo-', label='Original')
plt.plot(rotated[:,0], rotated[:,1], 'go-', label='Rotated')
plt.title("Rotation (45°)")
plt.legend()
plt.axis('equal')
plt.grid()

# All together
plt.subplot(1,3,3)
plt.plot(rectangle[:,0], rectangle[:,1], 'bo-', label='Original')
plt.plot(scaled[:,0], scaled[:,1], 'ro-', label='Scaled')
plt.plot(rotated[:,0], rotated[:,1], 'go-', label='Rotated')
plt.title("Comparison")
plt.legend()
plt.axis('equal')
plt.grid()

plt.suptitle("Rectangle Transformations")
plt.show()



# PART B: MNIST TRANSFORMATIONS

print("\n=== PART B: MNIST DIGIT TRANSFORMATIONS ===")

# Load dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Select one digit
image = x_train[0]

# Convert to coordinate points
coords = np.column_stack(np.where(image > 0))


# Rotation (45 degrees)

rotated_coords = coords @ R.T


# Reflection (Y-axis)

reflect = np.array([[-1, 0],
                    [0, 1]])

reflected_coords = coords @ reflect.T


# Plot MNIST Transformations

plt.figure(figsize=(12,4))

# Original
plt.subplot(1,3,1)
plt.scatter(coords[:,1], coords[:,0], s=1)
plt.title(f"Original Digit: {y_train[0]}")
plt.gca().invert_yaxis()
plt.axis('equal')

# Rotated
plt.subplot(1,3,2)
plt.scatter(rotated_coords[:,1], rotated_coords[:,0], s=1)
plt.title("Rotated 45°")
plt.gca().invert_yaxis()
plt.axis('equal')

# Reflected
plt.subplot(1,3,3)
plt.scatter(reflected_coords[:,1], coords[:,0], s=1)
plt.title("Reflected (Y-axis)")
plt.gca().invert_yaxis()
plt.axis('equal')

plt.suptitle("MNIST Digit Transformations")
plt.show()



# FINAL EXPLANATION OUTPUT


print("""
SUMMARY:
- Scaling changes the size of shapes (stretch/shrink).
- Rotation changes orientation but keeps shape unchanged.
- Reflection flips the shape (mirror effect).

In machine learning:
- These transformations help models recognize patterns
  even when orientation or position changes.
""")
