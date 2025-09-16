import torch
import torch.nn as nn

class BoundaryEncoder(nn.Module):
    def __init__(self, in_dim=3, k=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.ReLU(),
        )
        self.fc_mean = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, k))
        self.fc_var  = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, k))

    def forward(self, x_bn3):            # x_bn3: [B, N, 3]
        assert x_bn3.dim() == 3, f"expected [B,N,3], got {x_bn3.shape}"
        x_b3n = x_bn3.transpose(1, 2)    # -> [B, 3, N]
        h = self.conv(x_b3n)             # [B, 512, N]
        g = torch.max(h, dim=2).values   # [B, 512]
        mu = self.fc_mean(g)             # [B, k]
        lv = self.fc_var(g)              # [B, k]
        return mu, lv

class BoundaryDecoder(nn.Module):
    """MLP: z[k] -> N*3 coordinates."""
    def __init__(self, k=32, N=400):
        super().__init__()
        self.N = N
        self.mlp = nn.Sequential(
            nn.Linear(k, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, N*3)
        )
    def forward(self, z_bk):               # [B,k]
        out = self.mlp(z_bk)               # [B, N*3]
        return out.view(-1, self.N, 3)     # [B,N,3]

class BoundaryVAE(nn.Module):
    def __init__(self, in_dim=3, k=32, N=400):
        super().__init__()
        self.enc = BoundaryEncoder(in_dim=in_dim, k=k)
        self.dec = BoundaryDecoder(k=k, N=N)



    def forward(self, x_bn3):  # [B, N, 3]
        print( x_bn3.shape)
        mu, lv = self.enc(x_bn3)  # [B, k], [B, k]
        std = torch.exp(0.5 * lv)
        z = mu + torch.randn_like(std) * std
        x_hat = self.dec(z)  # [B, N, 3]
        return x_hat, mu, lv

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu                       # use mean at eval

    def forward(self, x_nb3):
        mu, logvar = self.enc(x_nb3)        # [1,k] each
        z = self.reparameterize(mu, logvar, self.training)  # [1,k]
        x_hat = self.dec(z)                  # [1,N,3]
        return x_hat.squeeze(0), mu.squeeze(0), logvar.squeeze(0)
def chamfer_l2(a_bn3, b_bn3):
    # a,b: [B,N,3]
    d = torch.cdist(a_bn3, b_bn3, p=2)**2        # [B,N,N]
    a2b = d.min(dim=2).values.mean(dim=1)        # [B]
    b2a = d.min(dim=1).values.mean(dim=1)        # [B]
    return (a2b + b2a).mean()

# ---------- example ----------
N, k = 400, 32
model = BoundaryVAE(in_dim=3, k=k, N=N)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
beta = 0.1


import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader


class WallClouds(Dataset):
    def __init__(self, folder="carotid_geom_npz", N=400, split="train", seed=0):
        paths = sorted(glob.glob(os.path.join(folder, "case_*.npz")))
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)
        n = len(paths)
        if split == "train":
            self.paths = paths[:int(0.8*n)]
        elif split == "val":
            self.paths = paths[int(0.8*n):int(0.9*n)]
        else:
            self.paths = paths[int(0.9*n):]
        self.N = N
        self.rng = np.random.default_rng(seed + (0 if split=="train" else 1))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X = np.load(self.paths[idx])["X_wall"].astype(np.float32)  # [M,3], M varies
        M = X.shape[0]
        # resample to fixed N
        if M >= self.N:
            sel = self.rng.choice(M, size=self.N, replace=False)

        else:
            sel = self.rng.choice(M, size=self.N, replace=True)
        pts = X[sel]  # [N,3]
        # per-shape normalize: center to mean, scale to unit radius

        ctr = pts.mean(0, keepdims=True)
        pts = pts - ctr
        scale = np.max(np.linalg.norm(pts, axis=1))
        pts = pts / (scale + 1e-6)
        return torch.from_numpy(pts)   # [N,3]


from torch.utils.data import DataLoader

def train_vae(
    data_folder="carotid_geom_npz",
    N=400, k=32, beta=0.1,
    batch_size=16, epochs=60, lr=1e-3,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = WallClouds(data_folder, N=N, split="train", seed=0)
    ds_va = WallClouds(data_folder, N=N, split="val",   seed=0)
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BoundaryVAE(k=k, N=N).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tr_running = 0.0
        for x in tr:
            x = x.to(device)                         # [B,N,3]
            x_hat, mu, lv = model(x)                 # [B,N,3], [B,k], [B,k]
            rec = chamfer_l2(x_hat, x)
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
                rec = chamfer_l2(x_hat, x)
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
        data_folder="carotid_geom_npz",
        N=400, k=32, beta=0.1,
        batch_size=16, epochs=60, lr=1e-3
    )

