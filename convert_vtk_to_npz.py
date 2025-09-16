# pip install pyvista numpy
import numpy as np
import pyvista as pv
from pathlib import Path

def vtk_to_npz_dir(in_dir, out_dir, flow_name="flow", dtype=np.float32, pattern=("*.vtp","*.vtk")):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for pat in pattern:
        files += sorted(in_dir.glob(pat))
    assert files, f"No input files found in {in_dir} with {pattern}"

    manifest = []
    for f in files:
        m = pv.read(f.as_posix())
        P = np.asarray(m.points)                          # (N,3)
        if flow_name not in m.point_data:
            raise KeyError(f"{f.name}: point_data has {list(m.point_data.keys())}, missing '{flow_name}'")
        V = np.asarray(m.point_data[flow_name])           # (N,3)
        if P.shape[0] != V.shape[0] or P.shape[1]!=3 or V.shape[1]!=3:
            raise ValueError(f"{f.name}: unexpected shapes {P.shape}, {V.shape}")

        P = P.astype(dtype, copy=False)
        V = V.astype(dtype, copy=False)

        out = out_dir / (f.stem + ".npz")
        np.savez(out, xyz=P, uvw=V)                      # keys: 'xyz', 'uvw'
        manifest.append(out.as_posix())

    # write a manifest (plain text list of paths)
    (out_dir / "manifest.txt").write_text("\n".join(manifest))
    print(f"Saved {len(manifest)} npz files to {out_dir}")
    return manifest


vtk_to_npz_dir("augmented_inlets", "npz_inlets")
