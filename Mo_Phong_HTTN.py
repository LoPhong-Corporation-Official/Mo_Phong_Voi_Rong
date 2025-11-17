"""A SCRIPT CREATED BY LOPHONG CORPORATION"""
#To use this script you need these libraries:
#1.numpy
#2.matplotlib
#You can get it by this command: pip install numpy matplotlib
#THANKS FOR CHOOSING LOPHONG CORPORATION
#Copyright LoPhong Corporation 2018-2025
#Turn dreams into reality

#NHẬP THƯ VIỆN
"""import numpy as np
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
plt.show()"""
"""
tornado_2d3d_pytorch.py
- Mô phỏng tornado đồng bộ 3D + 2D (top-down) realtime
- Tính toán bằng PyTorch (dùng GPU nếu available), vẽ bằng Matplotlib (CPU)
- Sửa lỗi: deb_z initialization, đúng kiểu cho torch.sin/torch.clamp, hiệu năng hợp lý
"""

import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# -----------------------
# Device (GPU AMD via ROCm if available)
# -----------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# -----------------------
# Simulation parameters
# -----------------------
# Particles (vortex)
N = 4000                 # số hạt vòi rồng
H = 30.0                 # chiều cao vòi rồng
R_base = 8.0             # bán kính đáy
R_top = 1.8              # bán kính đỉnh

# Debris near ground
DEBRIS = 900

# 2D wind grid (top-down)
GRID = 36

# dynamics
base_swirl = 0.12
swirl_amp = 0.06
rise_speed = 0.06

# -----------------------
# Helper: convert tensor -> numpy for plotting
# -----------------------
def to_cpu_np(tensor):
    """Chuyển tensor (CPU/GPU) về numpy float32"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return np.array(tensor)

# -----------------------
# Initialize vortex particles (on device)
# -----------------------
torch.manual_seed(42)

z = torch.rand(N, device=device) * H                      # z height (fixed)
radius = R_base - (R_base - R_top) * (z / H)              # radius per particle
theta = torch.rand(N, device=device) * 2 * math.pi       # angular position

# initial x,y
x = radius * torch.cos(theta)
y = radius * torch.sin(theta)

# per-particle radial distance (used when updating)
r_particle = radius.clone()

# -----------------------
# Initialize debris (near ground)
# -----------------------
deb_x = (torch.rand(DEBRIS, device=device) - 0.5) * (R_base * 2.0)
deb_y = (torch.rand(DEBRIS, device=device) - 0.5) * (R_base * 2.0)
deb_z = torch.zeros(DEBRIS, device=device)   # <-- ensure defined (fix NameError)

# -----------------------
# 2D wind grid (device)
# -----------------------
gx = torch.linspace(-10.0, 10.0, GRID, device=device)
gy = torch.linspace(-10.0, 10.0, GRID, device=device)
GX, GY = torch.meshgrid(gx, gy, indexing="xy")
density_base = torch.exp(-(GX**2 + GY**2)/40.0)

# -----------------------
# Wind field function (accepts scalar t as torch.Tensor or float)
# Returns (U,V,strength) on device
# -----------------------
def wind_field(t):
    # Accept either tensor or float
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(float(t), dtype=torch.float32, device=device)
    dx = -(GY) / (GX**2 + GY**2 + 2.0)
    dy =  (GX) / (GX**2 + GY**2 + 2.0)
    strength = 1.3 + 0.8 * torch.sin(t * 0.09)    # tensor op
    return dx * strength, dy * strength, strength

# -----------------------
# Matplotlib setup (single fig with 3D + 2D + colorbar)
# -----------------------
fig = plt.figure(figsize=(14, 7))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(222)
axbar = fig.add_subplot(224)

# 3D axes limits
ax3d.set_xlim(-10, 10)
ax3d.set_ylim(-10, 10)
ax3d.set_zlim(0, H)
ax3d.set_title("3D Tornado (PyTorch sim)")
ax3d.set_box_aspect((1,1,0.6))

# initial scatter (pass placeholder arrays)
sc_vortex = ax3d.scatter(
    to_cpu_np(x[:200]), to_cpu_np(y[:200]), to_cpu_np(z[:200]),
    s=1.5, c=to_cpu_np(z[:200]), cmap='gray', alpha=0.9
)
sc_debris = ax3d.scatter(
    to_cpu_np(deb_x), to_cpu_np(deb_y), to_cpu_np(deb_z),
    s=8, c='sienna', alpha=0.9
)

# 2D wind field initial
U0, V0, S0 = wind_field(0.0)
U0c = to_cpu_np(U0); V0c = to_cpu_np(V0)

Q = ax2d.quiver(
    to_cpu_np(GX), to_cpu_np(GY),
    U0c, V0c, scale=40, pivot='mid', color='white', alpha=0.8
)
density_img = ax2d.imshow(
    to_cpu_np(density_base),
    cmap='inferno',
    alpha=0.6,
    extent=[-10,10,-10,10],
    origin='lower'
)
ax2d.set_facecolor("k")
ax2d.set_title("2D Wind Field (Top-down)")
ax2d.set_xlim(-10,10); ax2d.set_ylim(-10,10)

# colorbar for wind speed (block-ish)
wind_levels = [0,10,20,30,40,50,60,70,80,100,130]
wind_colors = [
    "#9EE8FF","#5CCCCC","#C7E76E","#FFE66E","#FFB84D",
    "#FF8C33","#FF5C5C","#FF3333","#CC0033","#660099"
]
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(wind_colors)
norm = BoundaryNorm(wind_levels, ncolors=cmap.N)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                  cax=axbar, boundaries=wind_levels, ticks=wind_levels, orientation='vertical')
cb.set_label("Wind speed (km/h)")

plt.tight_layout()

# -----------------------
# Animation update function (core)
# - Uses torch on device for array ops
# - Converts to numpy for plotting only
# -----------------------
frame_dt = 1.0   # time step per frame (arbitrary units)

def update(frame):
    global x, y, theta, deb_x, deb_y, deb_z, r_particle

    t = torch.tensor(float(frame), dtype=torch.float32, device=device)

    # --- wind (tensor) ---
    U, V, strength = wind_field(t)   # arrays on device

    # --- tornado swirl: compute angular increment (tensor scalar)
    # Use tensor 't' for sin to keep ops on device
    ang = base_swirl + swirl_amp * torch.sin(t * 0.05)   # scalar tensor
    # expand ang to vector multiply
    # rotate points by angle 'ang' (same angle for all here)
    cs = torch.cos(ang); sn = torch.sin(ang)
    x_new = x * cs - y * sn
    y_new = x * sn + y * cs
    x = x_new; y = y_new

    # optionally add small radial drift upward (simulate rising)
    # move small amount outward/inward based on z
    radial_drift = 1e-3 * (0.5 - torch.rand_like(x))   # tiny jitter
    r_particle = torch.sqrt(x*x + y*y)
    x = x + radial_drift * (x / (r_particle + 1e-6))
    y = y + radial_drift * (y / (r_particle + 1e-6))

    # --- debris: noisy random walk + slight lift depending on time (scalar)
    # produce scalar lift between 0..3 clamped
    val = torch.abs(torch.sin(t * 0.06)) * 2.0
    lift = torch.clamp(val, 0.0, 3.0)    # scalar tensor
    # add Gaussian noise
    deb_x = deb_x + (0.3 * torch.randn_like(deb_x, device=device))
    deb_y = deb_y + (0.3 * torch.randn_like(deb_y, device=device))
    deb_z[:] = lift

    # --- update 2D density shown (use strength for visual) ---
    density = density_base * (0.6 + 0.5 * torch.abs(torch.sin(t * 0.07)))

    # --- Convert a subset or full arrays to numpy and update plots ---
    # For performance, draw subset of vortex points (sample)
    sample_step = max(1, N // 800)  # draw up to ~800 points
    idx = torch.arange(0, N, sample_step, device=device)

    x_cpu = to_cpu_np(x[idx])
    y_cpu = to_cpu_np(y[idx])
    z_cpu = to_cpu_np(z[idx])

    sc_vortex._offsets3d = (x_cpu, y_cpu, z_cpu)

    deb_x_cpu = to_cpu_np(deb_x)
    deb_y_cpu = to_cpu_np(deb_y)
    deb_z_cpu = to_cpu_np(deb_z)
    sc_debris._offsets3d = (deb_x_cpu, deb_y_cpu, deb_z_cpu)

    # update 2D quiver and density image
    Uc = to_cpu_np(U); Vc = to_cpu_np(V)
    Q.set_UVC(Uc, Vc)
    density_img.set_data(to_cpu_np(density))

    return sc_vortex, sc_debris, Q, density_img

# -----------------------
# Run animation
# -----------------------
ani = FuncAnimation(fig, update, frames=np.arange(0, 10000), interval=25, blit=False)
plt.show()

