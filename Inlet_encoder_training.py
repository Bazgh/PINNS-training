import torch
import torch.nn as nn

class InletEncoder(nn.Module):
    def __init__(self, in_dim=6, k=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.ReLU(),
        )
        self.fc_mean = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, k))
        self.fc_var  = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, k))

    def forward(self, x_bn6):                 # x_bn6: [B, N, 6]
        # sanity checks
        assert x_bn6.dim() == 3, f"expected 3D tensor [B,N,6], got {x_bn6.shape}"
        assert x_bn6.size(-1) == 6, f"last dim must be 6 (xyz+uvw), got {x_bn6.shape}"

        x_b6n = x_bn6.transpose(1, 2)         # [B, 6, N] -> channels-first for Conv1d
        h = self.conv(x_b6n)                   # [B, 512, N]
        g = h.max(dim=2).values                # [B, 512] (global max pool)
        mu = self.fc_mean(g)                   # [B, k]
        logvar = self.fc_var(g)                # [B, k]
        return mu, logvar


class InletDecoder(nn.Module):
    """MLP: z[k] -> N*6 coordinates."""
    def __init__(self, k=32, N=400):
        super().__init__()
        self.N = N
        self.mlp = nn.Sequential(
            nn.Linear(k, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, N*6)
        )
    def forward(self, z_bk):               # [B,k]
        out = self.mlp(z_bk)               # [B, N*6]
        return out.view(-1, self.N, 6)     # [B,N,6]

class InletVAE(nn.Module):
    def __init__(self, k=32, N=400):
        super().__init__()
        self.enc = InletEncoder(k=k)
        self.dec = InletDecoder(k=k, N=N)

    def forward(self, x_bn6):  # [B, N, 6]
        mu, lv = self.enc(x_bn6)  # [B, k], [B, k]
        std = torch.exp(0.5 * lv)
        z = mu + torch.randn_like(std) * std
        x_hat = self.dec(z)  # [B, N, 6]
        return x_hat, mu, lv



import torch

def chamfer_inlet_with_velocity(x_hat, x, w_pos=1.0, w_vel=1.0):
    """
    x_hat: [B, Np, 6]  -> [:, :, :3]=coords, [:, :, 3:]=vel
    x    : [B, Ng, 6]

    Returns: total_loss, coord_cd, vel_mismatch
    """
    xyz_pred, vel_pred = x_hat[..., :3], x_hat[..., 3:]
    xyz_gt,   vel_gt   = x[...,   :3], x[...,   3:]

    # --- Coordinate Chamfer (squared) ---
    d = torch.cdist(xyz_pred, xyz_gt, p=2)        # [B, Np, Ng]
    d2 = d.pow(2)

    # pred -> gt
    d2_p2g_vals, idx_p2g = d2.min(dim=2)          # [B, Np], [B, Np]
    # gt -> pred
    d2_g2p_vals, idx_g2p = d2.min(dim=1)          # [B, Ng], [B, Ng]

    cd_xyz = d2_p2g_vals.mean() + d2_g2p_vals.mean()

    # --- Velocity mismatch at those correspondences ---
    B = x_hat.shape[0]
    batch = torch.arange(B, device=x_hat.device)[:, None]

    # For each pred point, take the GT velocity at its nearest GT point
    vel_gt_for_pred = vel_gt[batch, idx_p2g]      # [B, Np, 3]
    # For each GT point, take the pred velocity at its nearest pred point
    vel_pred_for_gt = vel_pred[batch, idx_g2p]    # [B, Ng, 3]

    vel_p2g = (vel_pred - vel_gt_for_pred).pow(2).sum(-1).mean()
    vel_g2p = (vel_gt   - vel_pred_for_gt).pow(2).sum(-1).mean()
    vel_mismatch = vel_p2g + vel_g2p

    total = w_pos * cd_xyz + w_vel * vel_mismatch
    return total

# ---------- example ----------
N, k = 400, 32
model = InletVAE( k=k, N=N)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
beta = 0.1


import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

model2 = InletVAE(k=k, N=N)


class Inlet_coord_velocity(Dataset):
    def __init__(self, folder="npz_inlets", N=400, split="train", seed=0,
                 pattern="*.npz", normalize_coords=True, normalize_vel=False):
        # pick files
        all_paths = sorted(glob.glob(os.path.join(folder, pattern)))
        if not all_paths:
            raise FileNotFoundError(f"No files matched {os.path.join(folder, pattern)}")
        rng = np.random.default_rng(seed)
        rng.shuffle(all_paths)

        n = len(all_paths)
        if split == "train":
            self.paths = all_paths[: int(0.8 * n)]
        elif split == "val":
            self.paths = all_paths[int(0.8 * n): int(0.9 * n)]
        else:
            self.paths = all_paths[int(0.9 * n):]

        self.N = N
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        self.normalize_coords = normalize_coords
        self.normalize_vel = normalize_vel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        X = d["xyz"].astype(np.float32)   # [M,3]
        V = d["uvw"].astype(np.float32)   # [M,3]
        if X.shape != V.shape or X.shape[1] != 3:
            raise ValueError(f"{self.paths[idx]}: unexpected shapes xyz={X.shape}, uvw={V.shape}")

        M = X.shape[0]
        # resample to fixed N (same indices for coords and velocities)
        sel = (self.rng.choice(M, size=self.N, replace=False)
               if M >= self.N else
               self.rng.choice(M, size=self.N, replace=True))
        pts  = X[sel]     # [N,3]
        vels = V[sel]     # [N,3]

        # ---- per-sample normalization (optional) ----
        if self.normalize_coords:
            ctr = pts.mean(axis=0, keepdims=True)
            pts = pts - ctr
            radius = np.maximum(1e-6, np.linalg.norm(pts, axis=1).max())
            pts = pts / radius

        if self.normalize_vel:
            vmean = vels.mean(axis=0, keepdims=True)
            vstd  = vels.std(axis=0, keepdims=True)
            vels = (vels - vmean) / np.maximum(vstd, 1e-6)
            # (alternatively keep physical units: set normalize_vel=False)

        # concat in x,y,z,u,v,w order
        inlet_vector = np.concatenate([pts, vels], axis=1).astype(np.float32)  # [N,6]
        return torch.from_numpy(inlet_vector)


from torch.utils.data import DataLoader

def train_vae(
    data_folder="npz_inlets",
    N=400, k=32, beta=0.1,
    batch_size=16, epochs=60, lr=1e-3,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = Inlet_coord_velocity(data_folder, N=N, split="train", seed=0)
    ds_va = Inlet_coord_velocity(data_folder, N=N, split="val",   seed=0)
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    model = InletVAE(k=k, N=N).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tr_running = 0.0
        for x in tr:
            x = x.to(device)                         # [B,N,6]
            x_hat, mu, lv = model(x)                 # [B,N,6], [B,k], [B,k]
            rec = chamfer_inlet_with_velocity(x_hat, x, w_pos=1.0, w_vel=1.0)
            kl  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = rec + beta*kl
            opt.zero_grad(); loss.backward(); opt.step()
            tr_running += loss.item() * x.size(0)
        tr_loss = tr_running / len(ds_tr)

        # ---- val ----
        model.eval()
        va_running = 0.0
        with torch.no_grad():
            for x in va:
                x = x.to(device)
                x_hat, mu, lv = model(x)
                rec = chamfer_inlet_with_velocity(x_hat, x, w_pos=1.0, w_vel=1.0)
                kl  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = rec + beta*kl
                va_running += loss.item() * x.size(0)
        va_loss = va_running / len(ds_va)

        print(f"epoch {ep:03d} | train {tr_loss:.5f} | val {va_loss:.5f}")

        if va_loss < best_val:
            best_val = va_loss
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/geom_pointnet_vae_k{k}_N{N}.pt")
            print("  ↳ saved checkpoint")

    return model
if __name__ == "__main__":
    model = train_vae(
        data_folder="npz_inlets",
        N=400, k=32, beta=0.1,
        batch_size=16, epochs=60, lr=1e-3
    )


