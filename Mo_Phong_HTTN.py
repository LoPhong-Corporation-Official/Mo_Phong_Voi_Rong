"""A SCRIPT CREATED BY LOPHONG CORPORATION"""
#To use this script you need these libraries:
#1.numpy
#2.matplotlib
#You can get it by this command: pip install numpy matplotlib
#THANKS FOR CHOOSING LOPHONG CORPORATION
#Copyright LoPhong Corporation 2018-2025
#Turn dreams into reality

#NHẬP THƯ VIỆN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Tham số ---
n_particles = 500       # số phân tử xung quanh
height = 10
turns = 6
radius = 2
wind_strength = 0.5
wind_angle = np.pi/4
speed_factor = 0.1       # tốc độ phân tử

# --- Tạo vòi rồng trục xoắn ---
n_points = 1000
theta = np.linspace(0, 2*np.pi*turns, n_points)
z = np.linspace(0, height, n_points)
r = radius*(1 - z/height)
x = r*np.cos(theta)
y = r*np.sin(theta)

# --- Tạo các phân tử ngẫu nhiên xung quanh ---
particles = np.random.rand(n_particles,3)
particles[:,0] = (particles[:,0]-0.5)*4*radius
particles[:,1] = (particles[:,1]-0.5)*4*radius
particles[:,2] = particles[:,2]*height

# --- Animation update ---
def update(frame):
    ax3d.cla()
    ax2d.cla()
    
    # --- Vòi rồng 3D ---
    x_rot = x*np.cos(frame*0.05) - y*np.sin(frame*0.05)
    y_rot = x*np.sin(frame*0.05) + y*np.cos(frame*0.05)
    
    # --- Tác động gió ---
    wind_x = wind_strength*np.cos(wind_angle)*np.exp(-z/height)
    wind_y = wind_strength*np.sin(wind_angle)*np.exp(-z/height)
    
    ax3d.plot(x_rot + wind_x, y_rot + wind_y, z, color='blue', alpha=0.7)
    
    # --- Di chuyển phân tử theo xoắn + gió ---
    global particles
    particles[:,0] += -0.05*particles[:,1] + wind_x[::2][:n_particles]*speed_factor
    particles[:,1] += 0.05*particles[:,0] + wind_y[::2][:n_particles]*speed_factor
    particles[:,2] += 0.02*np.sin(frame*0.1)
    
    # Reset nếu phân tử vượt chiều cao
    particles[particles[:,2]>height,2] = 0
    
    # Vẽ phân tử 3D
    ax3d.scatter(particles[:,0], particles[:,1], particles[:,2], color='cyan', s=5)
    
    ax3d.set_xlim(-3,3)
    ax3d.set_ylim(-3,3)
    ax3d.set_zlim(0,height)
    ax3d.set_title("Vòi rồng 3D với phân tử & gió")
    
    # --- Vẽ luồng gió 2D (chiếu X-Y) ---
    ax2d.quiver(particles[:,0], particles[:,1], 
                -0.05*particles[:,1]+wind_x[::2][:n_particles]*speed_factor,
                0.05*particles[:,0]+wind_y[::2][:n_particles]*speed_factor,
                color='red', alpha=0.5)
    ax2d.set_xlim(-3,3)
    ax2d.set_ylim(-3,3)
    ax2d.set_title("Chiếu luồng gió 2D X-Y")

# --- Setup figure ---
fig = plt.figure(figsize=(12,6))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

anim = FuncAnimation(fig, update, frames=200, interval=50)
plt.show()

