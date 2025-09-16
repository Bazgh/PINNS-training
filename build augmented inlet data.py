# pip install pyvista numpy
import numpy as np
import pyvista as pv
from pathlib import Path

rng = np.random.default_rng(42)

# -------- Settings --------
N_SAMPLES = 300
OUT_DIR = Path("augmented_inlets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# noise strength as a fraction of natural variability
POS_SIGMA_FRAC = 0.25    # 0.25 * per-axis std of coords
VEL_SIGMA_FRAC = 0.25    # 0.25 * per-component std of velocity

# -------- Load base --------
mesh0 = pv.read("Inlet_velocity.vtk")
assert isinstance(mesh0, pv.PolyData), f"Expected PolyData, got {type(mesh0).__name__}"
P0 = np.asarray(mesh0.points)                   # (N,3)
V0 = np.asarray(mesh0.point_data["flow"])       # (N,3)
N = P0.shape[0]

# stats for scaling
pos_std = P0.std(axis=0, ddof=1)               # (3,)
vel_std = V0.std(axis=0, ddof=1)               # (3,)
pos_sigma = POS_SIGMA_FRAC * pos_std
vel_sigma = VEL_SIGMA_FRAC * vel_std

print("Per-axis coord std:", pos_std)
print("Per-comp vel std  :", vel_std)
print("Using sigmas      :", pos_sigma, vel_sigma)

# -------- Generate --------
for i in range(1, N_SAMPLES + 1):
    # copy base
    m = mesh0.copy(deep=True)

    # Gaussian perturbations
    dP = rng.normal(loc=0.0, scale=pos_sigma, size=(N, 3))
    dV = rng.normal(loc=0.0, scale=vel_sigma, size=(N, 3))

    # apply
    m.points = P0 + dP
    V = V0 + dV
    m.point_data["flow"] = V

    # save
    out = OUT_DIR / f"inlet_{i:04d}.vtp"  # use .vtp (XML PolyData, robust)
    m.save(out.as_posix())

print(f"Done. Wrote {N_SAMPLES} files to {OUT_DIR.resolve()}")
