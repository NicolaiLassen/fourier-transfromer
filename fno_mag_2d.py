
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

### Landau-Lifshitz-Gilbert experiment FNO

# problem 4
f = h5py.File('./problem4.h5')
sample_idx = 0
prob_sample = np.array(f[str(sample_idx)]['sequence'])
prob_field = np.array( f[str(sample_idx)]['field'])

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
    def __init__(self, channels=3, modes_x=8, modes_y=8, dim=32, depth=16, mlp=128):
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
            nn.Linear(dim, mlp),
            nn.GELU(),
            nn.Linear(mlp, channels)
        )

    def forward(self, x, f):
        # b t c w h
        w, h = x.shape[3:]
        
        x = rearrange(x, "b t c w h -> b w h (t c)")
        x = normalize(x, dim=-1)

        grid = self.get_grid(x.shape, x.device)
        
        f = repeat(f, "b d -> b w h d", w=w, h=h)
        f = normalize(f, dim=-1)
        
        # Field and bound grid
        x = torch.cat((x, f, grid), dim=-1)
    
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

batch_size = 512

epochs = 1000
learning_rate = 0.001
scheduler_step = 10
scheduler_gamma = 0.5

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

from util.data_loader import read_h5_dataset

if __name__ == '__main__':
    plt.figure(figsize=(16,9),dpi=140)

    train_set = read_h5_dataset(
        "C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\field_s_state_train_circ_paper.h5",
        block_size,
        batch_size,
        stride,
        -1
    )

    dl = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                persistent_workers=True,
                num_workers=1,
                pin_memory=True,
            )

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
    # plot mag mean data
    ################################################################
        
    def plot_prediction(y_pred, y_target, epoch) -> None:
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
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='16')
        plt.ylabel('$M_i [-]$', fontsize=32)
        plt.xlabel('$Time [ns]$', fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.title('Fourier Neural Operator', fontsize=48)
        plt.savefig("./fno/images/fno_{}.png".format(epoch))
        plt.clf()
        
    ################################################################
    # train
    ################################################################
        
    myloss = LpLoss(size_average=False)

    from tqdm import tqdm
    pbar = tqdm(range(epochs))
    loss_epoch = 9999
    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0

        for step_idx, x in enumerate(dl):
            
            s = x["states"][:, :T_in].cuda()
            f = x["fields"].cuda()

            loss = 0
            yy = x["states"][:, T_in:T+T_in].cuda()

            for t in range(0, T, step):
                y = yy[:, t:t + step]
                
                im = model(s, f)

                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)

                s = torch.cat((s[:, step:], im), dim=1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'step': '{}/{}'.format(step_idx + 1,len(dl)), 'loss': loss_epoch})

        loss_epoch = train_l2_step / len(dl)
        
        scheduler.step()
        model.eval()
        
        with torch.no_grad():
            if ep % 10 == 10 - 1:
                y = torch.tensor(prob_sample).cuda().float()
                whole_seq = torch.tensor(prob_sample[:T_in]).unsqueeze(0).cuda().float()
                f = torch.tensor(prob_field)[:2].unsqueeze(0).cuda().float()
                
                for t in range(0, 400 - T_in, step):
                    im = model(whole_seq[:, t:], f.cuda())
                    whole_seq = torch.cat((whole_seq, im), dim=1)
                
                plot_prediction(whole_seq[0], y, ep + 1)
                torch.save(model.state_dict(), "./fno/ckpt/fno_{}.pth".format(ep + 1))