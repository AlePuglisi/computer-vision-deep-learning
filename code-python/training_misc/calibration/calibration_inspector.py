import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial.transform import Rotation
from PIL import Image

# Read camera poses
poses_df = pd.read_csv('./raspi_usb_cam/output/camera_poses.csv')
print("Camera Poses:")
print(poses_df)
print()

# Convert to camera positions in world frame
camera_positions = []
camera_orientations = []

for i in range(len(poses_df)):
    # Get rotation vector and translation vector
    rvec = np.array([poses_df.loc[i, 'rx'], 
                     poses_df.loc[i, 'ry'], 
                     poses_df.loc[i, 'rz']])
    tvec = np.array([poses_df.loc[i, 'tx'], 
                     poses_df.loc[i, 'ty'], 
                     poses_df.loc[i, 'tz']])
    
    # Convert rotation vector to rotation matrix
    R = Rotation.from_rotvec(rvec).as_matrix()

        
    # Camera center in world coordinates: C = -R^T * t
    C_world = -R.T @ tvec
    camera_positions.append(C_world)
    
    # Camera orientation in world frame
    R_world = R.T
    camera_orientations.append(R_world)

camera_positions = np.array(camera_positions)

# Extract positions
tx_world = camera_positions[:, 0]
ty_world = camera_positions[:, 1]
tz_world = camera_positions[:, 2]

# Visualize camera positions in 3D
fig = plt.figure(figsize=(15, 6))

# 3D plot of camera positions
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(tx_world, ty_world, tz_world, c='red', marker='o', s=100, label='Camera positions')
ax1.scatter(0, 0, 0, c='blue', marker='^', s=200, label='Chessboard origin')

# Draw lines from origin to each camera
for i in range(len(tx_world)):
    ax1.plot([0, tx_world[i]], [0, ty_world[i]], [0, tz_world[i]], 'gray', alpha=0.3)
    ax1.text(tx_world[i], ty_world[i], tz_world[i], f'{i}', fontsize=8)

# Draw chessboard axes
axis_length = 100
ax1.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.2, label='X (right)')
ax1.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.2, label='Y (down)')
ax1.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.2, label='Z (out)')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('Camera Positions in World Frame')
ax1.legend()
ax1.grid(True)

# 2D projection (X-Y view - top view)
ax2 = fig.add_subplot(132)
ax2.scatter(tx_world, ty_world, c='red', marker='o', s=100)
ax2.scatter(0, 0, c='blue', marker='^', s=200, label='Chessboard origin')

for i in range(len(tx_world)):
    ax2.plot([0, tx_world[i]], [0, ty_world[i]], 'gray', alpha=0.3)
    ax2.text(tx_world[i], ty_world[i], f'{i}', fontsize=10)

ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_title('Top View (X-Y plane)')
ax2.grid(True)
ax2.axis('equal')
ax2.legend()

# 2D projection (X-Z view - side view)
ax3 = fig.add_subplot(133)
ax3.scatter(tx_world, tz_world, c='red', marker='o', s=100)
ax3.scatter(0, 0, c='blue', marker='^', s=200, label='Chessboard origin')

for i in range(len(tx_world)):
    ax3.plot([0, tx_world[i]], [0, tz_world[i]], 'gray', alpha=0.3)
    ax3.text(tx_world[i], tz_world[i], f'{i}', fontsize=10)

ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Z (mm)')
ax3.set_title('Side View (X-Z plane)')
ax3.grid(True)
ax3.axis('equal')
ax3.legend()

plt.tight_layout()
plt.show()


# Plot camera positions and orientations
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define chessboard dimensions
num_squares_x = 11  # Number of squares along X
num_squares_y = 8  # Number of squares along Y
square_size = 22    # mm


checkerboard_width = num_squares_x * square_size
checkerboard_height = num_squares_y * square_size

# Function to draw chessboard
def draw_chessboard_3d(ax, num_x, num_y, square_size, z=0):
    """Draw a chessboard pattern using polygons"""
    
    for i in range(num_y):
        for j in range(num_x):
            # Determine if square should be black or white
            # Standard chessboard: top-left is black if (i+j) is even
            is_black = (i + j) % 2 == 0
            color = 'black' if is_black else 'white'
            
            # Define square corners
            x0 = j * square_size
            x1 = (j + 1) * square_size
            y0 = i * square_size
            y1 = (i + 1) * square_size
            
            # Create vertices for the square
            vertices = [
                [x0, y0, z],
                [x1, y0, z],
                [x1, y1, z],
                [x0, y1, z]
            ]
            
            # Create polygon
            poly = Poly3DCollection([vertices], alpha=0.9)
            poly.set_facecolor(color)
            poly.set_edgecolor('gray')
            poly.set_linewidth(0.5)
            ax.add_collection3d(poly)
    
    # Draw border
    border_vertices = [
        [0, 0, z],
        [checkerboard_width, 0, z],
        [checkerboard_width, checkerboard_height, z],
        [0, checkerboard_height, z]
    ]
    border = Poly3DCollection([border_vertices], alpha=0)
    border.set_edgecolor('black')
    border.set_linewidth(3)
    ax.add_collection3d(border)

# Draw the chessboard at Z=0
draw_chessboard_3d(ax, num_squares_x, num_squares_y, square_size, z=0)

# Plot chessboard frame
ax.scatter(0, 0, 0, c='blue', marker='^', s=200, label='Chessboard')
axis_length = 50
ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.05)

# Plot camera positions and orientations
for i in range(len(camera_positions)):
    pos = camera_positions[i]
    R_world = camera_orientations[i]
    
    # Plot camera position
    ax.scatter(pos[0], pos[1], pos[2], c='red', marker='o', s=50)
    
    # Camera axes (showing where camera is looking)
    axis_len = 40
    # Camera's X, Y, Z axes in world frame
    x_axis = R_world @ np.array([axis_len, 0, 0])
    y_axis = R_world @ np.array([0, axis_len, 0])
    z_axis = R_world @ np.array([0, 0, axis_len])  # This points in viewing direction
    
    ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], 
              color='red', alpha=0.6, arrow_length_ratio=0.3)
    ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], 
              color='green', alpha=0.6, arrow_length_ratio=0.3)
    ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], 
              color='blue', alpha=0.8, arrow_length_ratio=0.3, linewidth=2)
    
    ax.text(pos[0]+5, pos[1]+5, pos[2], f'{i}', fontsize=8)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Camera Positions and Orientations')
ax.legend()
ax.grid(True)

# Set equal aspect ratio
max_range = np.array([tx_world.max()-tx_world.min(), 
                      ty_world.max()-ty_world.min(), 
                      tz_world.max()-tz_world.min()]).max() / 2.0

mid_x = (tx_world.max()+tx_world.min()) * 0.5
mid_y = (ty_world.max()+ty_world.min()) * 0.5
mid_z = (tz_world.max()+tz_world.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ax.invert_zaxis()

plt.tight_layout()
plt.show()

# Print camera positions
print("\nCamera Centers in World Frame:")
for i in range(len(camera_positions)):
    print(f"Camera {i}: [{camera_positions[i, 0]:.2f}, {camera_positions[i, 1]:.2f}, {camera_positions[i, 2]:.2f}]")