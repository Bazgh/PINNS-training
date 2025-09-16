# make_geoms.py
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.interpolate import splprep, splev

# ---------- helpers ----------
def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n

def make_spline(ctrl, n=400, s=0.0):
    tck, _ = splprep(ctrl.T, s=s, k=min(3, len(ctrl)-1))
    u = np.linspace(0, 1, n)
    return np.stack(splev(u, tck), axis=1)

def orthonormal_frame(t):
    up = np.array([0,0,1.0]) if abs(t @ np.array([0,0,1.0])) < 0.95 else np.array([0,1.0,0])
    n1 = unit(np.cross(t, up)); n2 = unit(np.cross(t, n1))
    return n1, n2

def tube_surface(centerline, radii, n_s=200, n_theta=64):
    S = len(centerline)
    idx = np.linspace(0, S-1, n_s).astype(int)
    C  = centerline[idx]
    T  = unit(np.gradient(C, axis=0))
    wall = []
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    for i in range(n_s):
        n1, n2 = orthonormal_frame(T[i])
        ring = C[i] + radii[idx[i]]*(np.cos(thetas)[:,None]*n1 + np.sin(thetas)[:,None]*n2)
        wall.append(ring)
    return np.vstack(wall)  # [n_s*n_theta, 3]

# ---------- carotid generator ----------
@dataclass
class CarotidParams:
    D: float = 6.0
    L_cca: float = 40.0
    L_ica: float = 35.0
    L_eca: float = 30.0
    n_wall_s: int = 220
    n_wall_theta: int = 72

def sample_carotid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = CarotidParams()

    # randomize geometry a bit
    ang_ica = np.deg2rad(rng.uniform(20, 50))
    ang_eca = np.deg2rad(rng.uniform(-60, -30))
    L_cca   = rng.uniform(35, 45)
    L_ica   = rng.uniform(30, 40)
    L_eca   = rng.uniform(25, 35)

    # centerlines (Y-shape)
    cca_ctrl = np.array([[-L_cca,0,0], [-0.5*L_cca,0,0], [-0.2*L_cca,0,0], [0,0,0]], float)
    for j in (1,2):
        cca_ctrl[j] += rng.normal(scale=p.D*0.05, size=3)

    dir_ica = unit(np.array([np.cos(ang_ica), np.sin(ang_ica), 0.0]))
    dir_eca = unit(np.array([np.cos(ang_eca), np.sin(ang_eca), 0.0]))
    ica_ctrl = np.vstack([np.zeros(3), 0.4*L_ica*dir_ica, 0.8*L_ica*dir_ica, 1.0*L_ica*dir_ica])
    eca_ctrl = np.vstack([np.zeros(3), 0.4*L_eca*dir_eca, 0.8*L_eca*dir_eca, 1.0*L_eca*dir_eca])

    cca = make_spline(cca_ctrl, n=600)
    ica = make_spline(ica_ctrl, n=400)
    eca = make_spline(eca_ctrl, n=400)

    # radius along CCA with random stenosis
    s = np.linspace(0, 1, len(cca))
    base = (p.D/2)*(0.95 + 0.1*np.sin(2*np.pi*s))
    depth = rng.uniform(0.0, 0.5)
    pos   = rng.uniform(0.4, 0.8)
    width = rng.uniform(0.05, 0.12)
    sten  = 1.0 - depth*np.exp(-0.5*((s-pos)/width)**2)
    r_cca = base*sten
    r_ica = np.full(len(ica), p.D*0.45)
    r_eca = np.full(len(eca), p.D*0.40)

    # walls
    X_wall  = np.vstack([
        tube_surface(cca, r_cca, n_s=p.n_wall_s//2, n_theta=p.n_wall_theta),
        tube_surface(ica, r_ica, n_s=p.n_wall_s//4, n_theta=p.n_wall_theta),
        tube_surface(eca, r_eca, n_s=p.n_wall_s//4, n_theta=p.n_wall_theta),
    ])

    # normalize by D
    X_wall /= p.D
    return X_wall.astype(np.float32)

if __name__ == "__main__":
    out = Path("carotid_geom_npz"); out.mkdir(exist_ok=True)
    for j in range(1000):
        X_wall = sample_carotid(seed=1000 + j)
        np.savez_compressed(out / f"case_{j:04d}.npz", X_wall=X_wall)
    print("Saved 1000 cases to", out.resolve())

# for paraview
# convert_npz_to_vtp_with_verts.py
import numpy as np
import vtk
from pathlib import Path
import glob

src = Path("carotid_geom_npz")
dst = Path("carotid_geom_vtp"); dst.mkdir(exist_ok=True)

files = sorted(glob.glob(str(src / "case_*.npz")))
print(f"Found {len(files)} cases")

for f in files:
    X_wall = np.load(f)["X_wall"]  # [N,3]

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(X_wall))
    for i, (x, y, z) in enumerate(X_wall):
        points.SetPoint(i, float(x), float(y), float(z))

    # Create a vertex cell for each point so ParaView renders them as geometry
    verts = vtk.vtkCellArray()
    for i in range(len(X_wall)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(dst / (Path(f).stem + ".vtp")))
    writer.SetInputData(poly)
    writer.Write()
    print("saved", dst / (Path(f).stem + ".vtp"))
