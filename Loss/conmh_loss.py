import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')

from utils.tools import l2_norm


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1):
    out_1 = F.normalize(out_1, dim=1)
    out_2 = F.normalize(out_2, dim=1)

    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    if True:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss
    

def conmh_criterion(cfg, data, model, epoch, i, total_len, logger):
    data = {key: value.cuda() for key, value in data.items()}
    batchsize = data["visual_word"].size(0)
    device = data["visual_word"].device

    bool_masked_pos_1 = data["mask"][:,0,:].to(device, non_blocking=True).flatten(1).to(torch.bool)
    bool_masked_pos_2 = data["mask"][:,1,:].to(device, non_blocking=True).flatten(1).to(torch.bool)
        
    frame_1, hash_code_1 = model.forward(data["visual_word"], bool_masked_pos_1)
    frame_2, hash_code_2 = model.forward(data["visual_word"], bool_masked_pos_2)

    hash_code_1 = torch.mean(hash_code_1, 1)
    hash_code_2 = torch.mean(hash_code_2, 1)

    # recon_loss
    labels_1 = data["visual_word"][bool_masked_pos_1].reshape(batchsize, -1, cfg.feature_size)
    labels_2 = data["visual_word"][bool_masked_pos_2].reshape(batchsize, -1, cfg.feature_size)
    recon_loss = F.mse_loss(frame_1, labels_1) + F.mse_loss(frame_2, labels_2)

    # contra_loss
    contra_loss = dcl(hash_code_1, hash_code_2, batchsize, temperature=cfg.temperature, tau_plus=cfg.tau_plus)

    loss = recon_loss + cfg.a * contra_loss
    if i % 10 == 0 or batchsize < cfg.batch_size:  
        logger.info('Epoch:[%d/%d] Step:[%d/%d] reconstruction_loss: %.2f contra_loss: %.2f' \
            % (epoch+1, cfg.num_epochs, i, total_len,\
            recon_loss.data.cpu().numpy(), contra_loss.data.cpu().numpy()))

    return loss