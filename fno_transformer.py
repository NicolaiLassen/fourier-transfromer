from libs import *
from libs.ns_lite import *
get_seed(1127802)
import numpy as np

batch_size = 4

config = defaultdict(lambda: None,
                     node_feats=10+2,
                     pos_dim=2,
                     n_targets=1,
                     n_hidden=48, 
                     num_feat_layers=0,
                     num_encoder_layers=4,
                     n_head=1,
                     dim_feedforward=96,
                     attention_type='galerkin',
                     feat_extract_type=None,
                     xavier_init=0.01,
                     diagonal_weight=0.01,
                     layer_norm=True,
                     attn_norm=False,
                     return_attn_weight=False,
                     return_latent=False,
                     decoder_type='ifft',
                     freq_dim=20,
                     num_regressor_layers=2,
                     fourier_modes=12,
                     spacial_dim=2,
                     spacial_fc=False,
                     dropout=0.0,
                     encoder_dropout=0.0,
                     decoder_dropout=0.0,
                     ffn_dropout=0.05,
                     debug=False,
                     )

torch.cuda.empty_cache()
model = FourierTransformer2DLite(**config)
print(get_num_params(model))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

epochs = 100
lr = 1e-3
h = 1/64
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
""" scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, final_div_factor=1e4,
                       steps_per_epoch=len(train_loader), epochs=epochs)
"""


loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=0.1)

metric_func = WeightedL2Loss2d(regularizer=False, h=h)
