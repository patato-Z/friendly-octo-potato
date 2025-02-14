from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.module import Module
from utils.utils import get_model

from models.fs.no_selection import no_selection as no_selection_fs
from models.es.no_selection import no_selection as no_selection_es

class BaseModel(nn.Module):
    def __init__(self, args, backbone_model_name, fs, es, unique_values, features, mode=None):
        super(BaseModel, self).__init__()
        # embedding table
        self.embedding = nn.Embedding(sum(unique_values), embedding_dim = args.embedding_dim)
        torch.nn.init.normal_(self.embedding.weight.data, mean=0, std=0.01)
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))

        self.input_dims = args.embedding_dim * len(unique_values)

        self.bb = get_model(backbone_model_name, 'rec')(args, self.input_dims) # backbone model name
        self.fs = get_model(fs, 'fs')(args, unique_values, features) # feature selection method
        self.es = get_model(es, 'es')() # embedding search method
        self.args = args
        
        self.mode = mode


    def forward(self, x, current_epoch=0, current_step=0):
        raw_x = x.clone().detach()
            
        x = self.embedding(x + x.new_tensor(self.offsets))
        x = self.es(x)
        x = self.fs(x, current_epoch, current_step, raw_data = raw_x)
        x = self.bb(x)
        return x
    
    def set_optimizer(self):
        optimizer_bb = torch.optim.Adam([params for name,params in self.named_parameters() \
            if ('fs' not in name and 'es' not in name) or 'bb' in name], lr = self.args.learning_rate)
        
        if [params for name,params in self.named_parameters() if 'fs' in name] != []:
            optimizer_fs = torch.optim.Adam([params for name,params in self.named_parameters() \
                if 'fs' in name and 'bb' not in name], lr = self.args.learning_rate)
        else:
            optimizer_fs = None
        
        if [params for name,params in self.named_parameters() if 'es' in name] != []:
            optimizer_es = torch.optim.Adam([params for name,params in self.named_parameters() \
                if 'es' in name and 'bb' not in name], lr = self.args.learning_rate)
        else:
            optimizer_es = None
        return {'optimizer_bb': optimizer_bb, 'optimizer_fs': optimizer_fs, 'optimizer_es': optimizer_es}
    
    def set_optimizer_stage(self, args):
        assert isinstance(self.fs, no_selection_fs)
        assert isinstance(self.es, no_selection_es)
        
        if args.training_stage == '1':
            optim_mode = args.optim_stage_1
            lr = args.lr_stage_1
        elif args.training_stage == '2':
            optim_mode = args.optim_stage_2
            lr = args.lr_stage_2
        else:
            raise ValueError(args.training_stage)
        
        if optim_mode == 'only_out':
            params = [params for name,params in self.named_parameters() if 'bb' in name and 'out' in name]
        elif optim_mode == 'only_dense':
            params = [params for name,params in self.named_parameters() if 'bb' in name]
        elif optim_mode == 'only_emb':
            params = [params for name,params in self.named_parameters() if 'emb' in name]
        elif optim_mode == 'full':
            params = [params for name,params in self.named_parameters() if 'bb' in name or 'emb' in name]
        else:
            raise ValueError(optim_mode)
        
        optimizer_bb = torch.optim.Adam(params, lr=lr)
        return {'optimizer_bb': optimizer_bb, 'optimizer_fs': None, 'optimizer_es': None}
