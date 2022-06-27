
from einops import rearrange
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
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
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
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(10 * 3 + 2 + 3, self.width)
        
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x, f):
        #  TODO 
        b, t, c, w, h = x.shape
        
        x = rearrange(x, "b t c w h -> b w h (t c)")
        # print(x.shape)
              
        grid = self.get_grid(x.shape, x.device)
        
        # print(grid.shape)
        # print(x.shape)
        
        f = f.repeat(b, w, h, 1)
        f = normalize(f, dim=1)
        
        x = torch.cat((x, f), dim=-1)
        
        x = torch.cat((x, grid), dim=-1)
        # print(x.shape)
        
        x = self.fc0(x)
        # print(x.shape)
        
        x = x.permute(0, 3, 1, 2)
        
        # print(x.shape)
        
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # print(x.shape)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # print(x.shape)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # print(x.shape)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        x = rearrange(x, "b w h c -> b 1 c w h")
        # print(x.shape)
        # exit(0)
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

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

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
stride = 4

for i in range(0,  seq.size(0) - block_size + 1, stride):
    data_seqs.append(seq[i: i + block_size].unsqueeze(0))

data = torch.cat(data_seqs , dim=0).float().cuda()

trainset = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

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
# plot mean data
################################################################
    
def plot_prediction(y_pred, y_target) -> None:
    y_pred = y_pred.detach().cpu().numpy()
    y_target = y_target.detach().cpu().numpy()
    
    timeline = np.arange(y_pred.shape[0]) * 4e-12 * 1e9
    
    plt.plot(timeline, np.mean(y_target[:,0].reshape(y_target.shape[0],-1), axis=1), 'r')
    plt.plot(timeline, np.mean(y_target[:,1].reshape(y_target.shape[0],-1), axis=1), 'g')
    plt.plot(timeline, np.mean(y_target[:,2].reshape(y_target.shape[0],-1), axis=1), 'b')

    plt.plot(timeline, np.mean(y_pred[:,0].reshape(y_pred.shape[0],-1), axis=1), 'rx')
    plt.plot(timeline, np.mean(y_pred[:,1].reshape(y_pred.shape[0],-1), axis=1), 'gx')
    plt.plot(timeline, np.mean(y_pred[:,2].reshape(y_pred.shape[0],-1), axis=1), 'bx')
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Mx MagTense'),
        Line2D([0], [0], color='green', lw=4, label='My MagTense'),
        Line2D([0], [0], color='blue', lw=4, label='Mz MagTense'),
        Line2D([0], [0], marker='x', color='red', label='Mx Model'),
        Line2D([0], [0], marker='x', color='green', label='My Model'),
        Line2D([0], [0], marker='x', color='blue', label='Mz Model'),
    ]
    
    plt.legend(handles=legend_elements)
    plt.setp(plt.gca().get_legend().get_texts())
    plt.ylabel('$M_i [-]$')
    plt.xlabel('$Time [ns]$')
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    plt.grid()
    plt.title('Fourier Neural Operator')
    plt.show()
    plt.close()
    
################################################################
# train
################################################################
    
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0

    for x in trainset:
        
        loss = 0
        xx = x[:, :T_in]
        yy = x[:, T_in:T+T_in]

        for t in range(0, T, step):
            y = yy[:, t:t + step]
            
            im = model(xx, field.unsqueeze(0))
                        
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)

            xx = torch.cat((xx[:, step:], im), dim=1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(loss)
    scheduler.step()
    model.eval()
    
    with torch.no_grad():
        if ep % 49 == 0:
            whole_seq = seq[:T_in].unsqueeze(0).cuda()
            
            for t in range(0, 400 - T_in, step):
                im = model(whole_seq[:, t:], field.unsqueeze(0))
                whole_seq = torch.cat((whole_seq, im), dim=1)
            
            plot_prediction(whole_seq[0], seq)
            