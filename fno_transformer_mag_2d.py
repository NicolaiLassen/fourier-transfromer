
from einops import rearrange, repeat
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from timeit import default_timer
from torch.optim import Adam

import torch.fft as fft
from torch.nn.init import xavier_normal_

torch.manual_seed(0)
np.random.seed(0)
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import h5py
import copy

### Landau-Lifshitz-Gilbert experiment FNO transformer

# problem 4
f = h5py.File('./problem4.h5')
sample_idx = 0
prob_sample = np.array(f[str(sample_idx)]['sequence'])
prob_field = np.array( f[str(sample_idx)]['field'])

################################################################
# layers
################################################################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim, 
                 modes=8,
                 dropout=0.1,
                 norm='ortho',
                 activation=nn.SiLU,
                 return_freq=False,         
        ):
        super(SpectralConv2d, self).__init__()
        '''
        Modified Zongyi Li's SpectralConv2d PyTorch 1.8.0 code
        using only real weights
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        '''
        
        modes_x, modes_y = pair(modes)
        self.modes_x = modes_x
        self.modes_y = modes_y
        
        self.return_freq = return_freq
        self.norm = norm

        self.activation = activation()
        self.shortcut = nn.Linear(in_dim, out_dim)
        self.dropout =  nn.Dropout(dropout)

        self.scale = (1 / (in_dim * out_dim))
    
        self.fourier_weight = nn.ParameterList([nn.Parameter(
            self.scale * torch.FloatTensor(in_dim, out_dim, self.modes_x, self.modes_y, 2)) for _ in range(2)])
        
        for param in self.fourier_weight:
            xavier_normal_(param, gain=1/(in_dim*out_dim)
                           * np.sqrt(in_dim+out_dim))
        
    @staticmethod
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        b, _, w, h = x.shape
        
        res = self.shortcut(x)
        x = self.dropout(x)
        
        x = rearrange(x, "b w h c -> b c w h")
        
        # compute fourier coeffcients up to factor of e^(- constant)
        x_ft = fft.rfft2(x, s=(w, h), norm=self.norm)

        # multiply relevant Fourier modes
        out_ft = torch.zeros(b, self.out_channels,  w, h//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.fourier_weight[0])
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.fourier_weight[1])

        # return to physical space
        x = fft.irfft2(out_ft, s=(w, h), norm=self.norm)
        x = rearrange(x, "b c h w -> b w h c")
        
        x = self.activation(x + res)        
        
        if self.return_freq:
            return x, out_ft
        else:
            return x

class SpectralRegressor(nn.Module):
    def __init__(self, 
                 in_dim,
                 freq_dim,
                 out_dim,
                 activation=nn.SiLU,
                 normalizer=None,
                 return_freq=False,
                 dropout=0.1,
                 num_spectral_layers=4,
                 modes=8,
                 dim_fc=128
                 ):
        super(SpectralRegressor, self).__init__()

        self.spectral_conv = nn.ModuleList([SpectralConv2d(in_dim=in_dim,
                                                        out_dim=freq_dim,
                                                        modes=modes,
                                                        dropout=dropout,
                                                        activation=activation,
                                                        return_freq=return_freq,
                                                        )])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(SpectralConv2d(in_dim=freq_dim,
                                            out_dim=freq_dim,
                                            modes=modes,
                                            dropout=dropout,
                                            activation=activation,
                                            return_freq=return_freq,
                                            ))

        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, dim_fc),
            nn.SiLU(),
            nn.Linear(dim_fc, out_dim)
        )

    def forward(self, x, f):        
        for layer in self.spectral_conv:
                x = layer(x)

        x = self.regressor(x)
        return x

class GalerkinTransformer(nn.Module):
    def __init__(self, 
                 dim=64,
                 channels=3,
                 modes_x=8,
                 modes_y=8,
                 regressor_depth=4,
                 out_scale=2
                 ):
        super(GalerkinTransformer, self).__init__()

        self.regressor = SpectralRegressor(dim)
        
        self.dropout = nn.Dropout(self.dropout)
        self._get_encoder()

    def forward(self, x, f, pos, weight=None):
        # b t c w h
        
        x = self.dropout(x)
        
        for encoder in self.encoder_layers:
            x = encoder(x, pos, weight)

        x = self.regressor(x)
        return x
    
    def _get_encoder(self):
        encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                       n_head=self.n_head,
                                                       dim_feedforward=self.dim_feedforward,
                                                       layer_norm=self.layer_norm,
                                                       attention_type=self.attention_type,
                                                       attn_norm=self.attn_norm,
                                                       norm_type=self.norm_type,
                                                       xavier_init=self.xavier_init,
                                                       diagonal_weight=self.diagonal_weight,
                                                       dropout=self.encoder_dropout,
                                                       ffn_dropout=self.ffn_dropout,
                                                       pos_dim=self.pos_dim,
                                                       debug=self.debug)
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_grid(self, shape, device):
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

batch_size = 256

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

    model = GalerkinTransformer().cuda()
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
        plt.savefig("./galerkin/images/galerkin_{}.png".format(epoch))
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
                torch.save(model.state_dict(), "./galerkin/ckpt/galerkin_{}.pth".format(ep + 1))