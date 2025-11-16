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
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
#                TORNADO PARAMETERS
# =========================================================
N = 2600
H = 30
R_base = 7.5
R_top  = 1.8
wind_scale = 1.25

debris_N = 700

# =========================================================
#                INIT PARTICLES
# =========================================================
z = np.random.rand(N) * H
radius = R_base - (R_base - R_top) * (z / H)
theta = np.random.rand(N) * 2 * np.pi

x = radius * np.cos(theta)
y = radius * np.sin(theta)

# debris near ground
deb_x = np.random.uniform(-R_base, R_base, debris_N)
deb_y = np.random.uniform(-R_base, R_base, debris_N)
deb_z = np.zeros(debris_N)

# =========================================================
#                2D WIND FIELD
# =========================================================
grid_N = 25
gx = np.linspace(-10, 10, grid_N)
gy = np.linspace(-10, 10, grid_N)

GX, GY = np.meshgrid(gx, gy)

def wind_field(t):
    dx = -(GY) / (GX**2 + GY**2 + 2)
    dy = (GX) / (GX**2 + GY**2 + 2)

    strength = 1.3 + 0.7 * np.sin(t * 0.09)
    return dx * strength, dy * strength, strength

density_base = np.exp(-(GX**2 + GY**2)/40)

# =========================================================
#                WIND SPEED COLORBAR (KM/H)
# =========================================================
wind_levels = [0,10,20,30,40,50,60,70,80,100,130]
wind_colors = [
    "#9EE8FF", "#5CCCCC", "#C7E76E", "#FFE66E", "#FFB84D",
    "#FF8C33", "#FF5C5C", "#FF3333", "#CC0033", "#660099"
]

cmap = ListedColormap(wind_colors)
norm = BoundaryNorm(wind_levels, cmap.N)

# =========================================================
#                FIGURE LAYOUT
# =========================================================
fig = plt.figure(figsize=(14, 7))

ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(222)
axbar = fig.add_subplot(224)

# ---------------- 3D VIEW ----------------
ax3d.set_xlim(-10, 10)
ax3d.set_ylim(-10, 10)
ax3d.set_zlim(0, H)
ax3d.set_title("3D Tornado (Realtime Synchronized)")
ax3d.set_axis_off()

scatter = ax3d.scatter(x, y, z, s=2, c=z, cmap="gray")
debris = ax3d.scatter(deb_x, deb_y, deb_z, s=6, c='brown')

# ---------------- 2D WIND FIELD ----------------
U, V, S = wind_field(0)

Q = ax2d.quiver(GX, GY, U, V, color="white")
density_img = ax2d.imshow(
    density_base,
    cmap="gray",
    alpha=0.55,
    extent=[-10,10,-10,10],
    origin='lower'
)

ax2d.set_facecolor("black")
ax2d.set_title("2D Wind Field (Top-down)")
ax2d.set_xlim(-10, 10)
ax2d.set_ylim(-10, 10)

# ---------------- COLORBAR ----------------
windbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axbar,
    boundaries=wind_levels,
    ticks=wind_levels,
    orientation="vertical"
)

windbar.set_label("Wind Speed (km/h)", fontsize=12)
axbar.remove()

# =========================================================
#                REALTIME UPDATE LOOP
# =========================================================
def update(frame):
    global theta, x, y

    # --- Wind ---
    U, V, S = wind_field(frame)

    # === 3D TORNADO ROTATION ===
    swirl = wind_scale * (0.4 + z/H)
    theta += swirl * 0.08

    radius = R_base - (R_base - R_top)*(z/H)
    x[:] = radius * np.cos(theta)
    y[:] = radius * np.sin(theta)

    scatter._offsets3d = (x, y, z)

    # === DEBRIS ===
    dtheta = swirl[:debris_N] * 1.4
    ang = np.arctan2(deb_y, deb_x) + dtheta
    r = np.sqrt(deb_x**2 + deb_y**2)

    deb_x[:] = r * np.cos(ang)
    deb_y[:] = r * np.sin(ang)

    debris._offsets3d = (deb_x, deb_y, deb_z)

    # === 2D VECTOR FIELD ===
    Q.set_UVC(U, V)

    # Air density pulsing
    density2 = density_base * (0.6 + 0.4*np.sin(frame * 0.07))
    density_img.set_data(density2)

    return scatter, debris, Q, density_img

# =========================================================
#                RUN ANIMATION
# =========================================================
ani = FuncAnimation(fig, update, interval=20)
plt.tight_layout()
plt.show()
