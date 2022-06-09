from tqdm import tqdm
import numpy as np
import h5py
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tavit import TAViT

# quick train setup

if __name__ == '__main__':
    seq = []
    block_size = 16
    stride = 2
    n_data = -1
    
    with h5py.File('fluid_dataset.h5', 'r') as f: 
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

    model = TAViT().cuda()
    
    trainset = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    
    trainset_count = len(trainset)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    pbar = tqdm(range(500))
    for epoch in pbar:
        running_loss = 0
        model.train()
        for x in trainset:

            optimizer.zero_grad()
            
            x = x.cuda()
            xi = x[:, :-1]
            xo = x[:, 1:]
            
            x_e, x_h, x_p, x_h_p = model(xi)
            
            loss = (
                0.5 * (F.mse_loss(x_p, xi) + F.mse_loss(x_h_p, xo)) +
                F.mse_loss(x_e[:, 1:], x_h[:, :-1])
            )
        
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
        pbar.set_postfix({'loss': running_loss / trainset_count})
        
    torch.save(model.state_dict(), "fluid_model.pt")