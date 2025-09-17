import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from Geometry_encoder_training import BoundaryEncoder
import torch.optim as optim
import matplotlib.pyplot as plt
import vtk
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BoundaryEncoder()

ckpt = torch.load("ckpts/geom_pointnet_vae_k32_N400.pt", map_location=device)

# If your checkpoint is wrapped like {"model_state_dict": ...}
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]

# Keep only encoder params and drop the "enc." prefix
enc_sd = {k.replace("enc.", "", 1): v for k, v in ckpt.items() if k.startswith("enc.")}

# Now load strictly (all expected keys present)
model.load_state_dict(enc_sd, strict=True)
#model.eval()

import os, numpy as np, torch, vtk


x_wall = r"C:\Users\AICVI\Desktop\zahra\deepxde\wall.vtk"
assert os.path.exists(x_wall), f"File not found: {x_wall}"

# Legacy POLYDATA reader
reader = vtk.vtkPolyDataReader()
reader.SetFileName(x_wall)
reader.Update()

poly = reader.GetOutput()  # vtkPolyData
n_points = poly.GetNumberOfPoints()
print("n_points wall:", n_points)
assert n_points > 0, "No points read. Is the file path/format correct?"

coords = np.array([poly.GetPoint(i) for i in range(n_points)], dtype=np.float32)  # [N,3]
coords_t = torch.from_numpy(coords).unsqueeze(0)                      # [1,N,3]

with torch.no_grad():
    mu, lv = model(coords_t)  # [1, k], [1, k]
    std = torch.exp(0.5 * lv)
    eps = torch.randn_like(std)
    z = mu + eps * std  # [1, k]
    geom_latent = z



def train(device,x,y,z,xb,yb,zb,ub,vb,wb,geom_latent,batchsize, learning_rate, epochs, path, Flag_batch, Diff, rho,
              Flag_BC_exact, Lambda_BC,nPt, T,x_inlet,y_inlet,z_inlet,u_inlet,v_inlet,w_inlet):
    if (Flag_batch):
        x = torch.Tensor(x)#.to(device)
        y = torch.Tensor(y)#.to(device)
        z= torch.Tensor(z)#.to(device)
        B = x.size(0)  # batch size
        geom_latent_k = geom_latent.expand(B, -1)  # [B, 32]
        geom_latent_k=torch.Tensor(geom_latent_k)
        x_cat = torch.cat([x,y,z, geom_latent_k], dim=1)
        # dataset = TensorDataset(x,y)
        xb = torch.Tensor(xb)#.to(device)
        yb = torch.Tensor(yb)#.to(device)
        zb= torch.Tensor(zb)#.to(device)
        BB = xb.size(0)  # batch size
        geom_latent_bk = geom_latent.expand(BB, -1)  # [B, 32]
        geom_latent_bk = torch.Tensor(geom_latent_bk)

        xb_cat = torch.cat([xb,yb,zb, geom_latent_bk], dim=1)
        #ub = torch.Tensor(ub).to(device)
        #vb = torch.Tensor(vb).to(device)
        #wb = torch.Tensor(wb).to(device)
        # dist = torch.Tensor(dist).to(device)
        bIn = x_inlet.size(0)
        geom_latent_bIn = geom_latent.expand(bIn, -1)  # [B, 32]
        """
        x_inlet = torch.Tensor(x_inlet).to(device)
        y_inlet = torch.Tensor(y_inlet).to(device)
        z_inlet = torch.Tensor(z_inlet).to(device)
        u_inlet = torch.Tensor(u_inlet).to(device)
        v_inlet = torch.Tensor(v_inlet).to(device)
        w_inlet=torch.Tensor(w_inlet).to(device)
        """
        if (1):  # Cuda slower in double?
            pass
            """"
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            z= z.type(torch.cuda.FloatTensor)
            x_cat= torch.cat([x,y, z,geom_latent_k], dim=1)
            xb = xb.type(torch.cuda.FloatTensor)
            yb = yb.type(torch.cuda.FloatTensor)
            zb= zb.type(torch.cuda.FloatTensor)
            xb_cat = xb_cat.type(torch.cuda.FloatTensor)
            ub = ub.type(torch.cuda.FloatTensor)
            vb = vb.type(torch.cuda.FloatTensor)
            wb= wb.type(torch.cuda.FloatTensor)
            # dist = dist.type(torch.cuda.FloatTensor)
            x_inlet = x_inlet.type(torch.cuda.FloatTensor)
            y_inlet = y_inlet.type(torch.cuda.FloatTensor)
            z_inlet = z_inlet.type(torch.cuda.FloatTensor)
            u_inlet = u_inlet.type(torch.cuda.FloatTensor)
            v_inlet = v_inlet.type(torch.cuda.FloatTensor)
            w_inlet = w_inlet.type(torch.cuda.FloatTensor)
            """

        dataset = TensorDataset(x,y,z,geom_latent_k)
        # dataset_bc = TensorDataset(x,y,xb,yb,ub,vb,dist)

        # dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)
        # dataloader_bc = DataLoader(dataset_bc, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = False )
    else:
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        # t = torch.Tensor(t_in).to(device)
    # x_test =  torch.Tensor(x_test).to(device)
    # y_test  = torch.Tensor(y_test).to(device)

    h_nD = 64  # for BC net
    h_D = 128  # for distance net
    h_n = 128  # for u,v,w,p
    input_n = 3+32  # this is what our answer is a function of. In the original example 3 : x,y,scale

    class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)

    class MySquared(nn.Module):
        def __init__(self, inplace=True):
            super(MySquared, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            return torch.square(x)
    class Net1_dist(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net1_dist, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_D),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_D, h_D),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_D, h_D),
                # nn.ReLU(),
                Swish(),

                nn.Linear(h_D, h_D),

                # nn.ReLU(),
                Swish(),

                nn.Linear(h_D, h_D),

                # nn.ReLU(),
                Swish(),

                nn.Linear(h_D, h_D),

                Swish(),
                nn.Linear(h_D, h_D),

                Swish(),
                nn.Linear(h_D, h_D),

                Swish(),
                nn.Linear(h_D, h_D),

                nn.Linear(h_D, 1),

                # nn.ReLU(), # make sure output is positive (does not work with PINN!)
                # nn.Sigmoid(), # make sure output is positive
                MySquared(),
            )

        # This function defines the forward rule of
        # output respect to input.
        def forward(self, x):
            output = self.main(x)
            return output

    class Net1_bc_u(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net1_bc_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_nD),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_nD, h_nD),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                # nn.ReLU(),
                Swish(),

                nn.Linear(h_nD, 1),

                nn.ReLU(),  # make sure output is positive
            )

        # This function defines the forward rule of
        # output respect to input.
        def forward(self, x):
            output = self.main(x)
            return output

    class Net1_bc_v(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net1_bc_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_nD),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_nD, h_nD),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),

                nn.Linear(h_nD, 1),

                nn.ReLU(),  # make sure output is positive
            )

        # This function defines the forward rule of
        # output respect to input.
        def forward(self, x):
            output = self.main(x)
            return output
    class Net1_bc_w(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net1_bc_w, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_nD),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_nD, h_nD),
                # nn.ReLU(),
                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),
                nn.Linear(h_nD, h_nD),

                Swish(),

                nn.Linear(h_nD, 1),

                nn.ReLU(),  # make sure output is positive
            )

        # This function defines the forward rule of
        # output respect to input.
        def forward(self, x):
            output = self.main(x)
            return output
    class Net2_u(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net2_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),

                nn.Linear(h_n, 1),
            )

        # This function defines the forward rule of
        # output respect to input.
        # def forward(self,x):
        def forward(self, x):
            output = self.main(x)
            # output_bc = net1_bc_u(x)
            # output_dist = net1_dist(x)
            if (Flag_BC_exact):
                output = output * (x - xStart) * (y - yStart) * (y - yEnd) + U_BC_in + (y - yStart) * (
                            y - yEnd)*(z-zStart)* (z-zEnd) # modify output to satisfy BC automatically #PINN-transfer-learning-BC-20

            return output
    class Net2_v(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net2_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),

                nn.Linear(h_n, 1),
            )

        # This function defines the forward rule of
        # output respect to input.
        # def forward(self,x):
        def forward(self, x):
            output = self.main(x)
            # output_bc = net1_bc_v(x)
            # output_dist = net1_dist(x)
            if (Flag_BC_exact):
                output = output * (x - xStart) * (x - xEnd) * (y - yStart) * (y - yEnd) *(z-zStart)* (z-zEnd)+ (
                            -0.9 * x + 1.)  # modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
            # return  output * (y_in-yStart) * (y_in- yStart_up)
            # return output * dist_bc + v_bc
            # return output *output_dist * Dist_net_scale + output_bc
            return output

    class Net2_w(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net2_w, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),

                nn.Linear(h_n, 1),
            )

        # This function defines the forward rule of
        # output respect to input.
        # def forward(self,x):
        def forward(self, x):
            output = self.main(x)
            # output_bc = net1_bc_v(x)
            # output_dist = net1_dist(x)
            if (Flag_BC_exact):
                output = output * (x - xStart) * (x - xEnd) * (y - yStart) * (y - yEnd)*(z-zStart)* (z-zEnd) + (
                            -0.9 * x + 1.)  # modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
            # return  output * (y_in-yStart) * (y_in- yStart_up)
            # return output * dist_bc + v_bc
            # return output *output_dist * Dist_net_scale + output_bc
            return output

    class Net2_p(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super(Net2_p, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),
                Swish(),
                nn.Linear(h_n, h_n),
                # nn.Tanh(),
                # nn.Sigmoid(),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),
                nn.Linear(h_n, h_n),

                Swish(),

                nn.Linear(h_n, 1),
            )

        # This function defines the forward rule of
        # output respect to input.
        def forward(self, x):
            output = self.main(x)
            # print('shape of xnet',x.shape) #Resuklts: shape of xnet torch.Size([batchsize, 2])
            if (Flag_BC_exact):
                output = output * (x - xStart) * (x - xEnd) * (y - yStart) * (y - yEnd) + (
                            -0.9 * x + 1.)  # modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
            # return  (1-x[:,0]) * output[:,0]  #Enforce P=0 at x=1 #Shape of output torch.Size([batchsize, 1])
            return output

    ################################################################
    ###### Define the neural networks for u,v (velocity) and p (pressure) ##############
    net2_u = Net2_u()#.to(device)
    net2_v = Net2_v()#.to(device)
    net2_w = Net2_w()#.to(device)
    net2_p = Net2_p()#.to(device)

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    # use the modules apply function to recursively apply the initialization
    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_w.apply(init_normal)
    net2_p.apply(init_normal)

    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)
    optimizer_w = optim.Adam(net2_w.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10 ** -15)

    ###### Define the boundary conditions ##############
    def criterion_bc(xb, yb,zb,geom_latent_bk, value_u, value_v,value_w, x, y,z,geom_latent_k):
        # xb = torch.Tensor(xb).to(device)
        # yb = torch.Tensor(yb).to(device)
        # value_u = torch.Tensor(value_u ).to(device)
        # value_v = torch.Tensor(value_v ).to(device)

        net_in = torch.cat((xb, yb,zb,geom_latent_bk), 1)  # boundary data
        u_bc = Net1_bc_u(net_in)
        v_bc = Net1_bc_v(net_in)
        w_bc = Net1_bc_w(net_in)
        loss_f = nn.MSELoss()
        loss1 = loss_f(u_bc, value_u) + loss_f(v_bc, value_v)+ loss_f(w_bc, value_w)
        if (1):  # Make the soln satisfy Laplacian
            # x = torch.Tensor(x).to(device)
            # y = torch.Tensor(y).to(device)
            x.requires_grad = True
            y.requires_grad = True
            z.requires_grad = True
            net_in2 = torch.cat((x, y,z,geom_latent_k), 1)  # entire data
            u_bc2 = Net1_bc_u(net_in2)
            v_bc2 = Net1_bc_v(net_in2)
            w_bc2= Net1_bc_w(net_in2)
            u_bc2 = u_bc2.view(len(u_bc2), -1)
            v_bc2 = v_bc2.view(len(v_bc2), -1)
            w_bc2= w_bc2.view(len(w_bc2), -1)

            u_x = torch.autograd.grad(u_bc2, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
            v_x = torch.autograd.grad(v_bc2, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
            v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
            w_x=torch.autograd.grad(w_bc2, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
            w_xx=torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
            u_y = torch.autograd.grad(u_bc2, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
            u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
            v_y = torch.autograd.grad(v_bc2, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
            v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
            w_y=torch.autograd.grad(w_bc2, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
            w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
            u_z=torch.autograd.grad(u_bc2, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
            u_zz=torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
            v_z=torch.autograd.grad(v_bc2, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
            v_zz=torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
            w_z=torch.autograd.grad(w_bc2, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
            w_zz=torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

            loss2 = u_xx / (X_scale ** 2) + u_yy+ u_zz
            loss3 = v_xx / (X_scale ** 2) + v_yy+ v_zz
            loss4=w_xx/(X_scale**2) + w_yy+w_zz
            loss5 = loss_f(loss2, torch.zeros_like(loss2)) + loss_f(loss3, torch.zeros_like(loss3))+loss_f(loss4, torch.zeros_like(loss4))

        # regularize to reduce the high values in the interior
        # loss = loss1 + 0.01 * ( loss_f(u_bc,torch.zeros_like(loss1)) + loss_f(v_bc,torch.zeros_like(loss1)) )
        loss = 300 * loss1 + loss5
        return loss

###### Define the Navier-Stokes equations here ##############
    def criterion(x, y,z,geom_latent_k):

        # print (x)
        # x = torch.Tensor(x).to(device)
        # y = torch.Tensor(y).to(device)
        # t = torch.Tensor(t).to(device)

        # x = torch.FloatTensor(x).to(device)
        # x= torch.from_numpy(x).to(device)

        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        # t.requires_grad = True
        # u0 = u0.detach()
        # v0 = v0.detach()

        # net_in = torch.cat((x),1)
        net_in = torch.cat((x, y,z,geom_latent_k), 1)
        u = net2_u(net_in)
        u = u.view(len(u), -1)
        v = net2_v(net_in)
        v = v.view(len(v), -1)
        w = net2_w(net_in)
        w = w.view(len(w), -1)
        P = net2_p(net_in)
        P = P.view(len(P), -1)

        # u = u * t + V_IC #Enforce I.C???
        # v = v * t + V_IC #Enforce I.C???

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        u_z=torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        u_zz=torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        v_z=torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        v_zz=torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        w_x=torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_xx=torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_y=torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_yy=torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_z=torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        w_zz=w_y=torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        P_x = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        P_y = torch.autograd.grad(P, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        P_z = torch.autograd.grad(P, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        # u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
        # v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]

        XX_scale = U_scale * (X_scale ** 2)
        YY_scale = U_scale * (Y_scale ** 2)
        UU_scale = U_scale ** 2

        loss_2 = u * u_x / X_scale + v * u_y / Y_scale - Diff * (u_xx / XX_scale + u_yy / YY_scale) + 1 / rho * (
                    P_x / (X_scale * UU_scale))  # X-dir
        loss_1 = u * v_x / X_scale + v * v_y / Y_scale - Diff * (v_xx / XX_scale + v_yy / YY_scale) + 1 / rho * (
                    P_y / (Y_scale * UU_scale))  # Y-dir
        loss_3 = (u_x / X_scale + v_y / Y_scale)  # continuity

        # MSE LOSS
        loss_f = nn.MSELoss()

        # Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1, torch.zeros_like(loss_1)) + loss_f(loss_2, torch.zeros_like(loss_2)) + loss_f(loss_3,
                                                                                                            torch.zeros_like(
                                                                                                                loss_3))

        return loss

###### Define boundary condition loss ##############
    def Loss_BC(xb, yb,zb,geom_latent_bk, ub, vb,wb, xb_inlet, yb_inlet,zb_inlet,geom_latent_bIn, ub_inlet,vb_inlet,wb_inlet, x, y,z,geom_latent_k):
        # Stream function
        if (0):
            xb = torch.FloatTensor(xb).to(device)
            yb = torch.FloatTensor(yb).to(device)
            zb=torch.FloatTensor(zb).to(device)
            geom_latent_bk = torch.FloatTensor(geom_latent_bk).to(device)

            ub = torch.FloatTensor(ub).to(device)
            vb = torch.FloatTensor(vb).to(device)
            wb=torch.FloatTensor(wb).to(device)
        # tb = torch.FloatTensor(tb).to(device)
        # t_ic = torch.FloatTensor(t_ic).to(device)
        # u_ic = torch.FloatTensor(u_ic).to(device)
        # v_ic = torch.FloatTensor(v_ic).to(device)
        # t_ic =  torch.zeros_like(t)

        # xb.requires_grad = True
        # yb.requires_grad = True
        # xb_inlet.requires_grad = True
        # yb_inlet.requires_grad = True

        # net_in = torch.cat((xb),1)
        net_in1 = torch.cat((xb, yb,zb,geom_latent_bk), 1)
        out1_u = net2_u(net_in1)
        out1_v = net2_v(net_in1)
        out1_w=net2_w(net_in1)

        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = out1_v.view(len(out1_v), -1)
        out1_w = out1_w.view(len(out1_w), -1)

        net_in2 = torch.cat((xb_inlet, yb_inlet,zb_inlet,geom_latent_bIn), 1)
        out2_u = net2_u(net_in2)
        out2_v = net2_v(net_in2)
        out2_w = net2_w(net_in2)

        out2_u = out2_u.view(len(out2_u), -1)
        out2_v = out2_v.view(len(out2_v), -1)
        out2_w = out2_w.view(len(out2_w), -1)

        loss_f = nn.MSELoss()
        loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v))
        loss_inlet = loss_f(out2_u, ub_inlet) + loss_f(out2_v, torch.zeros_like(out2_v))

        return 1. * loss_noslip + loss_inlet

    # Main loop

    tic = time.time()

    if (Flag_pretrain):
        print('Reading (pretrain) functions first...')
        net2_u.load_state_dict(torch.load(path + "sten_u" + ".pt"))
        net2_v.load_state_dict(torch.load(path + "sten_v" + ".pt"))
        net2_w.load_state_dict(torch.load(path + "sten_w" + ".pt"))
        net2_p.load_state_dict(torch.load(path + "sten_p" + ".pt"))

    if (Flag_schedule):
        scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

    if (Flag_batch):  # This one uses dataloader

        for epoch in range(epochs):
            # for batch_idx, (x_in,y_in) in enumerate(dataloader):
            # for batch_idx, (x_in,y_in,xb_in,yb_in,ub_in,vb_in) in enumerate(dataloader):
            loss_eqn_tot = 0.
            loss_bc_tot = 0.
            n = 0
            for batch_idx, (x_in, y_in,z_in,geom_latent_k) in enumerate(dataloader):
                # net_in = torch.cat((x_in,y_in),1)
                # u_bc = net1_bc_u(net_in)
                # v_bc = net1_bc_v(net_in)
                # dist_bc = net1_dist(net_in)

                # net2_psi.zero_grad()
                net2_u.zero_grad()
                net2_v.zero_grad()
                net2_w.zero_grad()
                net2_p.zero_grad()
                loss_eqn = criterion(x_in, y_in,z_in,geom_latent_k)
                loss_bc = Loss_BC(xb, yb,zb,geom_latent_bk, ub, vb,wb, x_inlet, y_inlet,z_inlet,geom_latent_bIn, u_inlet,v_inlet,w_inlet, x, y,z,geom_latent_k)
                loss = loss_eqn + Lambda_BC * loss_bc
                loss.backward()
                optimizer_u.step()
                optimizer_v.step()
                # optimizer_psi.step()
                optimizer_p.step()
                loss_eqn_tot += loss_eqn.item()
                loss_bc_tot += loss_bc.item()
                n += 1
                if batch_idx % 40 == 0:
                    # loss_bc = Loss_BC(xb,yb,ub,vb) #causes out of memory issue for large data in cuda
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f}'.format(
                        epoch, batch_idx * len(x_in), len(dataloader.dataset),
                               100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item()))
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} '.format(
                #	epoch, batch_idx * len(x_in), len(dataloader.dataset),
                #	100. * batch_idx / len(dataloader), loss.item()))
            if (Flag_schedule):
                scheduler_u.step()
                scheduler_v.step()
                scheduler_w.step()
                scheduler_p.step()
            loss_eqn_tot = loss_eqn_tot / n
            loss_bc_tot = loss_bc_tot / n
            print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot))
            print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])

        if (0):  # This causes out of memory in cuda in autodiff
            loss_eqn = criterion(x, y)
            loss_bc = Loss_BC(xb, yb, ub, vb)
            loss = loss_eqn  # + Lambda_BC* loss_bc
            print('**** Final (all batches) \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
                loss.item(), loss_bc.item()))

    else:
        for epoch in range(epochs):
            # zero gradient
            # net1.zero_grad()
            ##Closure function for LBFGS loop:
            # def closure():
            net2.zero_grad()
            loss_eqn = criterion(x, y)
            loss_bc = Loss_BC(xb, yb, cb)
            if (Flag_BC_exact):
                loss = loss_eqn  # + loss_bc
            else:
                loss = loss_eqn + Lambda_BC * loss_bc
            loss.backward()
            # return loss
            # loss = closure()
            # optimizer2.step(closure)
            # optimizer3.step(closure)
            # optimizer4.step(closure)
            optimizer_u.step()
            optimizer_v.step()
            optimizer_w.step()
            optimizer_p.step()
            if epoch % 10 == 0:
                print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
                    epoch, loss.item(), loss_bc.item()))

    toc = time.time()
    elapseTime = toc - tic
    print("elapse time in parallel = ", elapseTime)
    ###################
    net2_u.eval()
    net2_v.eval()
    net2_w.eval()
    net2_p.eval()
    # plot
    if (1):  # save network
        # torch.save(net2_psi.state_dict(),path+"bwd2len2streamf_step_psi_"+str(epochs)+".pt")
        torch.save(net2_p.state_dict(), path + "sten_p" + ".pt")
        torch.save(net2_u.state_dict(), path + "sten_u" + ".pt")
        torch.save(net2_v.state_dict(), path + "sten_v" + ".pt")
        torch.save(net2_w.state_dict(), path + "sten_w" + ".pt")
        # torch.save(net1_bc_u.state_dict(),path+"bwd2len2_step_bcu_"+str(epochs)+".pt")
        # torch.save(net1_bc_v.state_dict(),path+"bwd2len2_step_bcv_"+str(epochs)+".pt")
        # torch.save(net1_dist.state_dict(),path+"bwd2len2_step_dist_"+str(epochs)+".pt")
        print("Data saved!")

    net_in = torch.cat((x.requires_grad_(), y.requires_grad_(),z.requires_grad_(),geom_latent_k), 1)
    output_u = net2_u(net_in)  # evaluate model (runs out of memory for large GPU problems!)
    output_v = net2_v(net_in)  # evaluate model
    output_w = net2_w(net_in)

    output_u = output_u.cpu().data.numpy()  # need to convert to cpu before converting to numpy
    output_v = output_v.cpu().data.numpy()
    output_w = output_w.cpu().data.numpy()
    x = x.cpu()
    y = y.cpu()
    z = z.cpu()
    geom_latent_k = geom_latent_k.cpu()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(x.detach().numpy(), y.detach().numpy(),z.detach().numpy(),geom_latent_k.detach().numpy(), c=output_u, cmap='rainbow')
    plt.title('NN results, u')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(x.detach().numpy(), y.detach().numpy(),z.detach().numpy(),geom_latent_k.detach().numpy(), c=output_v, cmap='rainbow')
    plt.title('NN results, v')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(x.detach().numpy(), y.detach().numpy(),z.detach().numpy(),geom_latent_k.detach().numpy(), c=output_w, cmap='rainbow')
    plt.title('NN results, w')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.show()

    return




X_scale = 2.0  # The length of the  domain (need longer length for separation region)
Y_scale = 1.0  # 0.3
U_scale = 1.0
U_BC_in = 0.5
Flag_batch = True  # False #USe batch or not  #With batch getting error...
Flag_BC_exact = False  # If True enforces BC exactly HELPS ALOT!!! Not implemented in 2D
Lambda_BC = 20  # 50. #5. # If not enforcing BC exacctly, then this should be high


mesh_file = "internal.vtu"
bc_file_in =  "inlet_velocity.vtk"
bc_file_wall =  "wall.vtk"

batchsize = 16  # 256 seems faster on gpu
learning_rate = 1e-5  # 1e-4 / 5.  / 2.

epochs = 50

Flag_pretrain = False  # True #If true reads the nets from last run

Diff = 0.001
rho = 1.
T = 0.5  # total duraction
# nPt_time = 50 #number of time-steps

Flag_x_length = True  # if True scales the eqn such that the length of the domain is = X_scale
X_scale = 2.0  # The length of the  domain (need longer length for separation region)
Y_scale = 1.0  # 0.3
U_scale = 1.0
U_BC_in = 0.5

Lambda_div = 1.  # 10. #penalty factor for continuity eqn (Makes it worse!?)
Lambda_v = 1.  # 100. #10. #penalty factor for y-momentum equation

# https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
Flag_schedule = True  # If true change the learning rate
if (Flag_schedule):
    learning_rate = 5e-4  # starting learning rate
    step_epoch = 1200
    decay_rate = 0.1

if (not Flag_x_length):
    X_scale = 1.
    Y_scale = 1.

print('Loading', mesh_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print('n_points of the mesh:', n_points)
x_vtk_mesh = np.zeros((n_points, 1))
y_vtk_mesh = np.zeros((n_points, 1))
z_vtk_mesh = np.zeros((n_points, 1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
    pt_iso = data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt_iso[0]
    y_vtk_mesh[i] = pt_iso[1]
    z_vtk_mesh[i] = pt_iso[2]
    VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))
y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))
z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))

nPt = 130  # 400 #130
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.0
zStart = 0.
zEnd = 1.
delta_circ = 0.2

t = np.linspace(0., T, nPt * nPt)
t = t.reshape(-1, 1)
print('shape of x', x.shape)
print('shape of y', y.shape)
print('shape of z', z.shape)
# print('shape of t',t.shape)


## Define boundary points

print('Loading', bc_file_in)
reader = vtk.vtkPolyDataReader()
reader.SetFileName(bc_file_in)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print('n_points of at inlet', n_points)
x_vtk_mesh = np.zeros((n_points, 1))
y_vtk_mesh = np.zeros((n_points, 1))
z_vtk_mesh = np.zeros((n_points, 1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
    pt_iso = data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt_iso[0]
    y_vtk_mesh[i] = pt_iso[1]
    z_vtk_mesh[i] = pt_iso[2]
    VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_in = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))
yb_in = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))
zb_in = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))
xb_in=torch.Tensor(xb_in)
yb_in=torch.Tensor(yb_in)
zb_in=torch.Tensor(yb_in)
print('Loading', bc_file_wall)
reader = vtk.vtkPolyDataReader()
reader.SetFileName(bc_file_wall)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print('n_points of at wall', n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw, 1))
y_vtk_mesh = np.zeros((n_pointsw, 1))
z_vtk_mesh = np.zeros((n_pointsw, 1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
    pt_iso = data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt_iso[0]
    y_vtk_mesh[i] = pt_iso[1]
    z_vtk_mesh[i] = pt_iso[2]
    VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_wall = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))
yb_wall = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))
zb_wall = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))

# u_in_BC = np.linspace(U_BC_in, U_BC_in, n_points) #constant uniform BC
u_in_BC = (yb_in[:]) * (0.3 - yb_in[:]) / 0.0225 * U_BC_in  # parabolic

v_in_BC = np.linspace(0., 0., n_points)
w_in_BC = np.linspace(0., 0., n_points)
u_wall_BC = np.linspace(0., 0., n_pointsw)
v_wall_BC = np.linspace(0., 0., n_pointsw)
w_wall_BC = np.linspace(0., 0., n_pointsw)

# t_BC = np.linspace(0., T, nPt_BC)
# t_BC = np.linspace(0., T, nPt_time)

# tb = np.concatenate((t_BC, t_BC, t_BC), 0)
# xb = np.concatenate((xb_wall), 0)
# yb = np.concatenate((yb_wall), 0)
xb = xb_wall
yb = yb_wall
zb = zb_wall

# ub = np.concatenate((u_wall_BC), 0)
# vb = np.concatenate((v_wall_BC), 0)
ub = u_wall_BC
vb = v_wall_BC
wb = w_wall_BC

# xb_inlet = np.concatenate((xb_in), 0)
# yb_inlet = np.concatenate((yb_in), 0)
# ub_inlet = np.concatenate((u_in_BC), 0)
# vb_inlet = np.concatenate((v_in_BC), 0)

xb_inlet = xb_in
yb_inlet = yb_in
zb_inlet = zb_in

ub_inlet = u_in_BC
vb_inlet = v_in_BC
wb_inlet = v_in_BC

### Trying to set distance function with Dirichlet BC everywhere
# xb_dist = np.concatenate((xleft, xup,xrightw, xdown,xdown2,xright), 0)
# yb_dist = np.concatenate((yleft, yup,yrightw, ydown,ydown2,yright), 0)
####


# tb= tb.reshape(-1, 1) #need to reshape to get 2D array
xb = xb.reshape(-1, 1)  # need to reshape to get 2D array
yb = yb.reshape(-1, 1)  # need to reshape to get 2D array
zb = zb.reshape(-1, 1)  # need to reshape to get 2D array
ub = ub.reshape(-1, 1)  # need to reshape to get 2D array
vb = vb.reshape(-1, 1)  # need to reshape to get 2D array
wb = wb.reshape(-1, 1)  # need to reshape to get 2D array
xb_inlet = xb_inlet.reshape(-1, 1)  # need to reshape to get 2D array
#xb_inlet = torch.from_numpy(xb_inlet.reshape(-1, 1)).float()
yb_inlet = yb_inlet.reshape(-1, 1)  # need to reshape to get 2D array
zb_inlet = zb_inlet.reshape(-1, 1)  # need to reshape to get 2D array
ub_inlet = ub_inlet.reshape(-1, 1)  # need to reshape to get 2D array
vb_inlet = vb_inlet.reshape(-1, 1)  # need to reshape to get 2D array
wb_inlet = wb_inlet.reshape(-1, 1)  # need to reshape to get 2D array

print('shape of xb', xb.shape)
print('shape of yb', yb.shape)
print('shape of ub', ub.shape)

# print('xb is', xb)
# print('yb is', yb)
# print('xb_inlet is', xb_inlet)
# print('yb_inlet is', yb_inlet)
# print('ub is', ub)
# print('ub_inlet is', ub_inlet)


# V_IC = 0. #I.C. for all velocoties.
# t_IC = np.linspace(0., 0., nPt*nPt)
# u_IC = np.linspace(V_IC, V_IC, nPt*nPt)
# v_IC = np.linspace(V_IC, V_IC, nPt*nPt)
# t_IC= t_IC.reshape(-1, 1)
# u_IC= u_IC.reshape(-1, 1)
# v_IC= v_IC.reshape(-1, 1)


path = "Results/"
#train(device,x,y,z,xb,yb,zb,ub,vb,wb,geom_latent,batchsize, learning_rate, epochs, path, Flag_batch, Diff, rho,
              #Flag_BC_exact, Lambda_BC,nPt, T,xb_inlet,yb_inlet,zb_inlet,ub_inlet,vb_inlet,wb_inlet)
def main():
    # prepare tensors/args here (CPU tensors; move to GPU inside the loop)
    # x, y, z, geom_latent_k, batchsize, num_epochs = ...
    train(device, x, y, z, xb, yb, zb, ub, vb, wb, geom_latent, batchsize, learning_rate, epochs, path, Flag_batch,
          Diff, rho,
          Flag_BC_exact, Lambda_BC, nPt, T, xb_inlet, yb_inlet, zb_inlet, ub_inlet, vb_inlet, wb_inlet)

if __name__ == "__main__":
    # Windows uses spawn; this guard is REQUIRED
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # optional but explicit
    # mp.freeze_support()  # only needed for PyInstaller/EXE cases
    main()
