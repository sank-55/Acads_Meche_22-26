import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================================
# Unsymmetrical Biaxial Bending Analysis of Equal-Angle (L) Beam
# Author: Grok (Structural Mechanics + Scientific Python)
# ================================================

# Parameters - easily adjustable
L = 100.0          # Length of each leg
t = 10.0           # Thickness of each leg
nx = 4000           # Grid resolution (x) - fine for accuracy and visualization
ny = 4000           # Grid resolution (y)
Mx = 666000.0       # Bending moment about global x-axis (arbitrary consistent units)
My = 000.0        # Bending moment about global y-axis

# Generate fine structured mesh over L x L domain
x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)

# Boolean material mask for inverted L in top-right corner:
# - Horizontal top leg: full width, thickness t from top
# - Vertical right leg: full height, thickness t from right
# Overlap at corner is correctly handled (material present)
mask = ((Y >= L - t) | (X >= L - t)).astype(float)

# Numerical integration setup
dx = x[1] - x[0]
dy = y[1] - y[0]
dA = dx * dy

# Compute Area
A = np.sum(mask * dA)
print(f"Cross-sectional Area: {A:.2f}")

# Compute Centroid (first moments)
x_c = np.sum(mask * X * dA) / A
y_c = np.sum(mask * Y * dA) / A
print(f"Centroid (x_c, y_c): ({x_c:.2f}, {y_c:.2f})")

# Shifted coordinates relative to centroid
x_s = X - x_c
y_s = Y - y_c

# Second moments of area (about centroid)
Ixx = np.sum(mask * y_s**2 * dA)   # ∫ y² dA
Iyy = np.sum(mask * x_s**2 * dA)   # ∫ x² dA
Ixy = np.sum(mask * x_s * y_s * dA) # ∫ x y dA

print(f"Ixx: {Ixx:.2f}")
print(f"Iyy: {Iyy:.2f}")
print(f"Ixy: {Ixy:.2f}")

# Denominator for unsymmetrical bending formula
denom = Ixx * Iyy - Ixy**2
if abs(denom) < 1e-8:
    raise ValueError("Denominator near zero - numerical instability")

# Flexural (normal) stress σ_z using general unsymmetrical bending equation
# σ = [ (Iyy * Mx - Ixy * My) * y - (Ixx * My - Ixy * Mx) * x ] / (Ixx*Iyy - Ixy²)
# Sign convention: positive moments follow standard right-hand rule; negative sign for typical beam compression on positive side
sigma = - ((Iyy * Mx - Ixy * My) * y_s - (Ixx * My - Ixy * Mx) * x_s) / denom

# Mask stress to material only (NaN elsewhere)
sigma_masked = np.where(mask > 0.5, sigma, np.nan)

# Publication-quality Matplotlib settings
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
})

# Figure 1: Geometry with Centroid
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(mask, extent=[0, L, 0, L], origin='lower', cmap='gray_r', alpha=0.85)
ax1.plot(x_c, y_c, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='Centroid')
ax1.set_title('L-Section Geometry (Top-Right Inverted L)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
ax1.legend(loc='upper left')
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Material Mask')
plt.tight_layout()
fig1.savefig('l_section_geometry.png', dpi=300)
print("Figure 1 saved: l_section_geometry.png")

# Figure 2: Filled Contour Plot with Neutral Axis
fig2, ax2 = plt.subplots()
levels = 100
cs = ax2.contourf(X, Y, sigma_masked, levels=levels, cmap='turbo')
# Neutral axis (σ = 0)
ax2.contour(X, Y, sigma_masked, levels=[0], colors='black', linewidths=2.5, linestyles='--')
ax2.plot(x_c, y_c, 'ro', markersize=8, markeredgecolor='white')
ax2.set_title('Flexural Stress Contour Plot\n(Black dashed line: Neutral Axis)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')
cbar2 = plt.colorbar(cs, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Flexural Stress σ')
plt.tight_layout()
fig2.savefig('stress_contour.png', dpi=300)
print("Figure 2 saved: stress_contour.png")

# Figure 3: Stress Heatmap (imshow with interpolation)
fig3, ax3 = plt.subplots()
im3 = ax3.imshow(sigma_masked, extent=[0, L, 0, L], origin='lower',
                 cmap='turbo', interpolation='bilinear', vmin=np.nanmin(sigma_masked), vmax=np.nanmax(sigma_masked))
ax3.plot(x_c, y_c, 'ro', markersize=8, markeredgecolor='white')
ax3.set_title('Flexural Stress Heatmap')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_aspect('equal')
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Flexural Stress σ')
plt.tight_layout()
fig3.savefig('stress_heatmap.png', dpi=300)
print("Figure 3 saved: stress_heatmap.png")
