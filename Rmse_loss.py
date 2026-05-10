def rmse_loss(pred, targ):
    import torch
    import torch.nn.functional as F
    import numpy as np
    device = torch.device('cuda:0')
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(targ, np.ndarray):
        targ = torch.from_numpy(targ)

    pred = pred.reshape(-1, 1)
    targ = targ.reshape(-1, 1)
    pred = pred.to(device)
    targ = targ.to(device)
    denom = torch.mean(targ**2) 
    return torch.sqrt(F.mse_loss(pred,targ)/denom)
