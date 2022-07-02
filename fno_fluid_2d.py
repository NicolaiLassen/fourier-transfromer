from einops import rearrange, repeat
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from timeit import default_timer
from torch.optim import Adam

from torch.nn.functional import normalize

torch.manual_seed(0)
np.random.seed(0)
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import h5py

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, channels=1, modes_x=8, modes_y=8, dim=32, depth=8, out_scale=2):
        super(FNO2d, self).__init__()

        self.project = nn.LazyLinear(dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SpectralConv2d(dim, dim, modes_x, modes_y),
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim)
            ]))

        self.fc_out = nn.Sequential(
            nn.Linear(dim, dim * out_scale),
            nn.GELU(),
            nn.Linear(dim * out_scale, channels)
        )

    def forward(self, x):
        # b t c w h
        w, h = x.shape[3:]
        
        x = normalize(x, dim=0)
        x = rearrange(x, "b t c w h -> b w h (t c)")
              
        grid = self.get_grid(x.shape, x.device)
        
        # Field and bound grid
        x = torch.cat((x, grid), dim=-1)
    
        x = self.project(x)
        
        x = rearrange(x, "b w h c -> b c w h")
        
        for fo, w, bn in self.layers:
            x1 = fo(x)
            x2 = w(x)
            x = bn(x1 + x2)
            x = F.gelu(x)
        
        x = rearrange(x, "b c w h -> b w h c")
        x = self.fc_out(x)
        
        x = rearrange(x, "b w h c -> b 1 c w h")
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################

modes = 8
width = 64

batch_size = 8

epochs = 1000
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

sub = 1 
S = 64
T_in = 10
T = 10
step = 1

################################################################
# load data
################################################################

block_size = 20
data_seqs = []
stride = 20

if __name__ == '__main__':
    plt.figure(figsize=(16,9),dpi=140)

    seq = []
    block_size = 20
    stride = 20
    n_data = -1
    
    with h5py.File('./data/fluid_dataset.h5', 'r') as f: 
        n_seq = 0
        for key in f.keys():
            data_series = torch.Tensor(np.array(f[key]))
           
            for i in range(0,  data_series.size(0) - block_size + 1, stride):
                seq.append(data_series[i: i + block_size].unsqueeze(0))

            n_seq = n_seq + 1
            if(n_data > 0 and n_seq >= n_data):  # If we have enough time-series samples break loop
                break

    data = torch.cat(seq , dim=0).float().unsqueeze(2)
    
    train_len = int(len(data) * 0.9)
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_len, len(data) - train_len])

    trainset = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    ################################################################
    # training and evaluation
    ################################################################

    model = FNO2d().cuda()
    # model = torch.load()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    ################################################################
    # loss function
    ################################################################

    class LpLoss(object):
        def __init__(self, d=2, p=2, size_average=True, reduction=True):
            super(LpLoss, self).__init__()

            #Dimension and Lp-norm type are postive
            assert d > 0 and p > 0

            self.d = d
            self.p = p
            self.reduction = reduction
            self.size_average = size_average

        def abs(self, x, y):
            num_examples = x.size()[0]

            #Assume uniform mesh
            h = 1.0 / (x.size()[1] - 1.0)

            all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

            if self.reduction:
                if self.size_average:
                    return torch.mean(all_norms)
                else:
                    return torch.sum(all_norms)

            return all_norms

        def rel(self, x, y):
            num_examples = x.size()[0]

            diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

            if self.reduction:
                if self.size_average:
                    return torch.mean(diff_norms/y_norms)
                else:
                    return torch.sum(diff_norms/y_norms)

            return diff_norms/y_norms

        def __call__(self, x, y):
            return self.rel(x, y)
        
    ################################################################
    # train
    ################################################################
        
    myloss = LpLoss(size_average=False)

    from matplotlib.animation import FuncAnimation, writers
    from tqdm import tqdm
    pbar = tqdm(range(epochs))
    loss_epoch = 9999
    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0

        for step_idx, x in enumerate(trainset):
            
            s = x[:, :T_in].cuda()
            
            loss = 0
            yy = x[:, T_in:T+T_in].cuda()

            for t in range(0, T, step):
                y = yy[:, t:t + step]
                
                im = model(s)

                loss += myloss(torch.flatten(im, start_dim=1), torch.flatten(y, start_dim=1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)

                s = torch.cat((s[:, step:], im), dim=1)

            train_l2_step += loss.item()
            l2_full =myloss(torch.flatten(yy, start_dim=1), torch.flatten(pred, start_dim=1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'step': '{}/{}'.format(step_idx + 1,len(trainset)), 'loss': loss_epoch})

        loss_epoch = train_l2_step / len(trainset)
        
        scheduler.step()
        model.eval()
        
        with torch.no_grad():
            if ep % 10 == 10 - 1:
                
                y = next(iter(trainset)).cuda()
                whole_seq = y[:1, :T_in]                
                
                for t in range(0, 100 - T_in, step):
                    im = model(whole_seq[:, t:])
                    whole_seq = torch.cat((whole_seq, im), dim=1)
                
                fig = plt.figure()
                im = plt.imshow(whole_seq[0,0,0].detach().cpu().numpy(), vmin=0, vmax=1, cmap=plt.cm.gray,
                                interpolation='bicubic', animated=True, origin='lower')
                
                def animate(t):
                    im.set_data(whole_seq[0,t,0].detach().cpu().numpy(),)
                    return im,
                
                anim = FuncAnimation(fig, animate, frames=100, interval=60, blit=True)    
                
                anim.save('fluid_model.gif', fps=64, dpi=72, writer='imagemagick')
                
                # torch.save(model.state_dict(), "./fno/ckpt/fno_{}.pth".format(ep + 1))