# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:56:29 2025

@author: Bazghandi
"""

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import vtk

LATENT = 128

# ---------------- Trunk & Branch ----------------
class Trunk(nn.Module):
    # maps [B, 3] -> [B, 128]
    def __init__(self, in_dim=3, latent=LATENT, width=128, depth=4):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, latent)]
        self.net = nn.Sequential(*layers)
    def forward(self, x_b3):            # x_b3: [B,3]
        #x = x_b3.squeeze(1)
        return self.net(x_b3)               # -> [B,128]
    
class Branch(nn.Module):
    # maps [Nc, 6] -> [Nc, 128]
    def __init__(self, in_dim=6, latent=LATENT, width=128, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, latent)]
        self.net = nn.Sequential(*layers)
    def forward(self, coarse_nc6):      # coarse_1nc6: [Nc,6]
        #c = coarse_1nc6.squeeze(0)       # -> [Nc,6]
        return self.net(coarse_nc6)               # -> [Nc,128]
# -------------- Scalar DeepONet head --------------
class ScalarDeepONet(nn.Module):
    """
    Produces a scalar field (e.g., U or V or W or P):
      trunk(grid[B,]) -> [B,128]
      branch(coarse[Nc,6]) -> [Nc,128]
      output: [B] with u_b = sum_i dot(T_b, B_i)
    """
    def __init__(self, trunk: Trunk, branch: Branch):
        super().__init__()
        self.trunk = trunk
        self.branch = branch

    def forward(self, grid_b3, coarse_1nc6):
        T = self.trunk(grid_b3)         # [B,128]
        Brc = self.branch(coarse_1nc6)   # [Nc,128]
        S = T @ Brc.T                    # [B,Nc]  (all dot products)
        out = S.sum(dim=1)               # [B]     (sum across Nc)
        return out
# -------------- Wrap four heads for U,V,W,P --------------
class DeepONetUVWP(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 separate networks; this shares nothing between them.
        self.trunk_u, self.branch_u = Trunk(), Branch()
        self.trunk_v, self.branch_v = Trunk(), Branch()
        self.trunk_w, self.branch_w = Trunk(), Branch()
        self.trunk_p, self.branch_p = Trunk(), Branch()

        self.head_u = ScalarDeepONet(self.trunk_u, self.branch_u)
        self.head_v = ScalarDeepONet(self.trunk_v, self.branch_v)
        self.head_w = ScalarDeepONet(self.trunk_w, self.branch_w)
        self.head_p = ScalarDeepONet(self.trunk_p, self.branch_p)

    def forward(self, grid_b3, coarse_nc6):
        U = self.head_u(grid_b3, coarse_nc6)  # [B]
        V = self.head_v(grid_b3, coarse_nc6)  # [B]
        W = self.head_w(grid_b3, coarse_nc6)  # [B]
        P = self.head_p(grid_b3, coarse_nc6)  # [B]
        return U, V, W, P
# ----------------- Autograd for gradients -----------------
def spatial_grads_scalar(scalar_B, grid_b3):
    """
    scalar_B: [B] (e.g., U, V, W, or P)
    grid_b13: [B,1,3] with requires_grad=True
    Returns partials d/dx, d/dy, d/dz as [B] each.
    """
    # Make sure grid has requires_grad=True before forward
    g = grad(scalar_B.sum(), grid_b3, create_graph=True)[0]  # [B,3]
    #g = g.squeeze(1)  # [B,3]
    dudx, dudy, dudz = g[:,0], g[:,1], g[:,2]
    return dudx, dudy, dudz

# Load sparse data
in_path = Path("sample_2.vtk")
reader = vtk.vtkPolyDataReader()
reader.SetFileName(str(in_path))
reader.Update()
pd = reader.GetOutput()

# --- Coordinates (N,3) ---
pts_vtk = pd.GetPoints().GetData()
coords = vtk_to_numpy(pts_vtk).astype(np.float64)   # (N,3)

# --- Velocities (N,3) from PointData["flow"] ---
arr = pd.GetPointData().GetArray("flow")
if arr is None:
    raise RuntimeError("PointData array 'flow' not found.")
vel = vtk_to_numpy(arr).astype(np.float64)          # (N,ncomp)
if vel.shape[1] > 3:                                # keep only 3 comps
    vel = vel[:, :3]

# --- Combine into (N,6): [x,y,z,u,v,w] ---
coarse_data=np.hstack([coords, vel])   # shape (N,6)

in_path = Path("wall.vtk")
reader = vtk.vtkPolyDataReader()
reader.SetFileName(str(in_path))
reader.Update()
pd = reader.GetOutput()

# Inspect number of points
print("Number of points:", pd.GetNumberOfPoints())

# Inspect all arrays stored in PointData
point_data = pd.GetPointData()
num_arrays = point_data.GetNumberOfArrays()
print("Available PointData arrays:", num_arrays)

for i in range(num_arrays):
    name = point_data.GetArrayName(i)
    print(f"Array {i}: {name}")

pts_vtk = pd.GetPoints().GetData()
wall_coords = vtk_to_numpy(pts_vtk).astype(np.float64)  # shape (N,3)
#load grid data
mesh_path = Path("internal.vtu")
ur = vtk.vtkXMLUnstructuredGridReader()
ur.SetFileName(str(mesh_path))
ur.Update()
ug = ur.GetOutput()
mesh_coords = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float32)  # (M,3)

print("Mesh points:", mesh_coords.shape)
mesh_coords= torch.from_numpy(mesh_coords)

import pyvista as pv

# Load the inlet dats
in_path = Path("inlet_velocity.vtk")
inlet = pv.read(in_path)

# Extract coordinates (N x 3)
coords_inlet = inlet.points.astype(np.float64)

# Extract velocity data stored in "flow" (N x 3)
flow = inlet.point_data["flow"].astype(np.float64)

# Split into u, v, w
ub_in = flow[:, 0]
vb_in= flow[:, 1]
wb_in = flow[:, 2]

# Quick check
print("Coords shape:", coords_inlet.shape)
print("Flow shape:", flow.shape)
print("u[0], v[0], w[0]:", ub_in[0], vb_in[0], wb_in[0])


u_wall=np.zeros(len(wall_coords.shape[0]))
v_wall=np.zeros(len(wall_coords.shape[0]))
w_wall=np.zeros(len(wall_coords.shape[0]))
#geo_train(device,x,y,xb,yb,ub,vb,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC,nPt,T,xb_inlet,yb_inlet,ub_inlet,vb_inlet )
def train(device,mesh_coords,wall_coords,u_wall=0,v_wall=0,w_wall=0,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC,nPt,T,coords_inlet,ub_inlet,vb_inlet,wb_inlet)
    if (Flag_batch):
        x= torch.Tensor(mesh_coords).to(device)
        # dataset = TensorDataset(x,y)
        xb= torch.Tensor(wall_coords).to(device)
        ub = torch.Tensor(u_wall).to(device)
        vb = torch.Tensor(v_wall).to(device)
        wb = torch.Tensor(w_wall).to(device)
        # dist = torch.Tensor(dist).to(device)
        xb_inlet = torch.Tensor(coords_inlet).to(device)
        ub_inlet = torch.Tensor(ub_inlet).to(device)
        vb_inlet = torch.Tensor(vb_inlet).to(device)
        wb_inlet = torch.Tensor(wb_inlet).to(device)
        if (1):  # Cuda slower in double?
            x = x.type(torch.cuda.FloatTensor)
            xb = xb.type(torch.cuda.FloatTensor)

            ub = ub.type(torch.cuda.FloatTensor)
            vb = vb.type(torch.cuda.FloatTensor)
            # dist = dist.type(torch.cuda.FloatTensor)
            xb_inlet = xb_inlet.type(torch.cuda.FloatTensor)
            ub_inlet = ub_inlet.type(torch.cuda.FloatTensor)
            vb_inlet = vb_inlet.type(torch.cuda.FloatTensor)
            dataset = TensorDataset(x)
            dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)
            model = DeepONetUVWP()
            U, V, W, P = model(dataloader, coarse_data)                 # each [B]

            Ux, Uy, Uz = spatial_grads_scalar(U, dataloader)
            Vx, Vy, Vz = spatial_grads_scalar(V, dataloader)
            Wx, Wy, Wz = spatial_grads_scalar(W, dataloader)
            Px, Py, Pz = spatial_grads_scalar(P,dataloader)

            ###### Define the boundary conditions ##############
            def criterion_bc(xb, ub_wall, vb_wall,wb_wall, x):
                # xb = torch.Tensor(xb).to(device)
                # yb = torch.Tensor(yb).to(device)
                # value_u = torch.Tensor(value_u ).to(device)
                # value_v = torch.Tensor(value_v ).to(device)


                u_bc = model(xb, coarse_data) [0]
                v_bc = model(xb, coarse_data) [1]
                w_bc = model(xb, coarse_data)[2]
                loss_f = nn.MSELoss()
                loss1 = loss_f(u_bc, ub_wall) + loss_f(v_bc,vb_wall)+loss_f(w_bc,wb_wall)
                if (1):  # Make the soln satisfy Laplacian
                    # x = torch.Tensor(x).to(device)
                    # y = torch.Tensor(y).to(device)
                    x[0].requires_grad = True
                    x[1].requires_grad = True
                    x[2].requires_grad = True
                    u_bc2 = model(x)[0]
                    v_bc2 = model(x)[1]
                    w_bc2 = model(x)[2]
                    u_bc2 = u_bc2.view(len(u_bc2), -1)
                    v_bc2 = v_bc2.view(len(v_bc2), -1)
                    w_bc2 = w_bc2.view(len(w_bc2), -1)

                    u_x = \
                    torch.autograd.grad(u_bc2, x[0], grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[
                        0]
                    u_xx = \
                    torch.autograd.grad(u_x, x[0], grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                    v_x = \
                    torch.autograd.grad(v_bc2, x[0], grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[
                        0]
                    v_xx = \
                    torch.autograd.grad(v_x, x[0], grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                    w_x=
                    w_xx=
                    u_y = \
                    torch.autograd.grad(u_bc2, x[1], grad_outputs=torch.ones_like(x[1]), create_graph=True, only_inputs=True)[
                        0]
                    u_yy = \
                    torch.autograd.grad(u_y, x[1], grad_outputs=torch.ones_like(x[1]), create_graph=True, only_inputs=True)[0]
                    v_y = \
                    torch.autograd.grad(v_bc2, x[1], grad_outputs=torch.ones_like(x[1]), create_graph=True, only_inputs=True)[
                        0]
                    v_yy = \
                    torch.autograd.grad(v_y, x[1], grad_outputs=torch.ones_like(x[1]), create_graph=True, only_inputs=True)[0]
                    w_y=
                    w_yy_
                    u_z=
                    u_zz=
                    v_z=
                    v_zz=
                    w_z=
                    w_zz
                    loss2 = u_xx / (X_scale ** 2) + u_yy+u_zz
                    loss3 = v_xx / (X_scale ** 2) + v_yy+v_zz
                    loss4= w_xx / (X_scale ** 2) + w_yy+w_zz
                    loss5= loss_f(loss2, torch.zeros_like(loss2)) + loss_f(loss3, torch.zeros_like(loss3))+loss_f(loss4, torch.zeros_like(loss4))

                # regularize to reduce the high values in the interior
                # loss = loss1 + 0.01 * ( loss_f(u_bc,torch.zeros_like(loss1)) + loss_f(v_bc,torch.zeros_like(loss1)) )
                loss = 300 * loss1 + loss5
                return loss