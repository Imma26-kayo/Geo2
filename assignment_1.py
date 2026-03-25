# Assignment 1
# Group 2

# Apart B soutions  
import open3d as o3d
import numpy as np
import random

# Load point cloud
pcd = o3d.io.read_point_cloud("bunny.xyz")

# Convert to numpy
points = np.asarray(pcd.points)

# Remove duplicates
points = np.unique(points, axis=0)

# Random sample up to 1000
if len(points) > 1000:
    points = points[np.random.choice(points.shape[0],1000,replace=False)]

pcd.points = o3d.utility.Vector3dVector(points)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])

# Select two points
p1 = points[0]
p2 = points[1]

# Vector
vector = p2 - p1

# Distance
distance = np.linalg.norm(vector)

# Midpoint
midpoint = (p1 + p2) / 2

print("Point1:",p1)
print("Point2:",p2)
print("Vector:",vector)
print("Distance:",distance)
print("Midpoint:",midpoint)

# Projection onto XY plane
points_xy = points.copy()
points_xy[:,2] = 0

pcd_xy = o3d.geometry.PointCloud()
pcd_xy.points = o3d.utility.Vector3dVector(points_xy)

o3d.visualization.draw_geometries([pcd_xy])
