import torch
import pdb

def conmh_inference(cfg, data, model):
    data = {key: value.cuda() for key, value in data.items()}

    my_H = model.inference(data["visual_word"])
    my_H = torch.mean(my_H, 1)
    
    BinaryCode = torch.sign(my_H)
    return BinaryCode