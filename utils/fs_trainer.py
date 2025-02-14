import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import nni
import datetime as dt
from utils.utils import EarlyStopper
from sklearn.metrics import roc_auc_score, log_loss
import logging
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pt = targets * inputs + (1 - targets) * (1 - inputs)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        F_loss = self.alpha * torch.pow(1 - pt, self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
            
    
    
class modeltrainer():
    def __init__(self, args, model, model_name, device, retrain):
        self.args = args
        if args.training_stage in ['1', '2']:
            self._set_args_stage()
        else:
            self.n_epoch = args.epoch
        self.model = model
        if args.training_stage is not None:
            self.optimizers = model.set_optimizer_stage(args) # dict of optimizers
        else:
            self.optimizers = model.set_optimizer() # dict of optimizers
        self.device = torch.device(device)
        self.model.to(self.device)
        if args.loss_func == 'bce':
            self.criterion = torch.nn.BCELoss(reduction='mean')
        elif args.loss_func == 'focal':
            self.criterion = FocalLoss(reduction='mean')
        else:
            raise ValueError(args.loss_func)
        self.save_path = None
        if args.save_mode == 'args':
            self.save_path = args.save_path
        elif args.save_mode == 'log':
            file_name = f"{args.save_path}.pth" if not args.save_path.endswith('.pth') else args.save_path
            self.save_path = os.path.join(args.log_path, file_name)
        elif args.save_mode is None:
            self.save_path = None
        else:
            raise ValueError(args.save_mode)
        
        if args.early_stop_mode == 'step':
            assert args.early_stop_epoch is not None   
            self.early_stopper_step = EarlyStopper(patience=args.patience_step, lr_min=args.lr_min, optims=self.optimizers)
        elif args.early_stop_mode == 'epoch':
            self.early_stopper = EarlyStopper(patience=args.patience, lr_min=args.lr_min, optims=self.optimizers)
        elif args.early_stop_mode in ['none', 'none_step']:
            self.early_stopper = None
        else:
            raise ValueError(args.early_stop_mode)
        self.retrain = retrain
        self.lambda_l2 = 0.

    def gat_loss(self, y_pred, y_true):

        if self.args.model == 'finalnet':
            return_dict = y_pred
            loss = self.criterion(return_dict["y_pred"], y_true)
            if self.model.bb.block_type == "2B":
                y1 = self.model.bb.output_activation(return_dict["y1"])
                y2 = self.model.bb.output_activation(return_dict["y2"])
                loss1 = self.criterion(y1, return_dict["y_pred"].detach())
                loss2 = self.criterion(y2, return_dict["y_pred"].detach())
                loss = loss + loss1 + loss2
            return loss
        else:
            
            return self.criterion(y_pred, y_true)

    
    def train_one_epoch(self, train_dataloader, val_dataloader, epoch_i, log_interval=10):
        
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        tk0 = tqdm.tqdm(train_dataloader, smoothing=0, mininterval=1.0)
        for i, batch_data_pack in enumerate(tk0):
            x, y, idx = batch_data_pack
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x, current_epoch=epoch_i, current_step=i)

            loss = self.gat_loss(y_pred, y.float().reshape(-1, 1))
            
            # 计算L2正则化项
            if self.lambda_l2 > 0.:
                l2_regularization = 0.
                params = []
                for g in self.optimizers['optimizer_bb'].param_groups:
                    params.extend(g['params'])
                for param in params:
                    l2_regularization += torch.norm(param, 2) ** 2
                l2_regularization *= self.lambda_l2
                loss += l2_regularization
            
            self.model.zero_grad()
            # self.optimizer.zero_grad()
            loss.backward()
            self.optimizers['optimizer_bb'].step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
            
            # TODO release中去掉 none_step
            if self.args.early_stop_mode == 'none':
                continue
            elif self.args.early_stop_mode == 'none_step':
                if i >= self.args.early_stop_start_step:
                    break
                else:
                    continue
            
            if val_dataloader and self.args.early_stop_mode == 'step' and epoch_i == self.args.early_stop_epoch and (i + 1) % self.args.early_stop_step_step == 0 and i >= self.args.early_stop_start_step:
                auc = self.evaluate(val_dataloader, i)
                logging.info(f'epoch: {epoch_i}')
                logging.info(f'step: {i}')
                logging.info(f'validation auc: {auc}')
                if self.early_stopper_step.stop_training(auc, self.model.state_dict()):
                    logging.info(f'validation: best auc: {self.early_stopper_step.best_auc}')
                    self.model.load_state_dict(self.early_stopper_step.best_weights)
                    break


    def fit(self, train_dataloader, val_dataloader=None):

        logging.info(f'training : early stop by {self.args.early_stop_mode}')
        
        all_start_time = dt.datetime.now()
        epoch_time_lis = []
        for epoch_i in range(self.n_epoch):
            if self.args.early_stop_mode == 'step' and epoch_i > self.args.early_stop_epoch:
                break
            # TODO release中去掉 none_step
            if self.args.early_stop_mode == 'none_step' and epoch_i > self.args.early_stop_epoch:
                break
            logging.info(f'epoch: {epoch_i}')
            epoch_start_time = dt.datetime.now()
            self.train_one_epoch(train_dataloader, val_dataloader, epoch_i)
            epoch_end_time = dt.datetime.now()
            epoch_time_lis.append((epoch_end_time - epoch_start_time).total_seconds())
            
            if self.args.early_stop_mode == 'none':
                continue
            
            if val_dataloader and self.args.early_stop_mode == 'epoch':
                auc = self.evaluate(val_dataloader, epoch_i)
                logging.info(f'epoch: {epoch_i}')
                logging.info(f'validation auc: {auc}')

                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    logging.info(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break

        all_end_time = dt.datetime.now()
        logging.info('all training time: {} s'.format((all_end_time - all_start_time).total_seconds()))
        logging.info('average epoch time: {} s'.format(sum(epoch_time_lis) / len(epoch_time_lis)))
        if self.save_path is not None:
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            logging.info(f"save at: {self.save_path}")
            torch.save(self.model.state_dict(), self.save_path)

    def _set_args_stage(self):
        ''' for two stage train, data retrieval '''
        if self.args.training_stage == '1':
            self.lambda_l2 = self.args.lambda_l2_stage_1
            self.n_epoch = self.args.epoch_stage_1 if self.args.epoch_stage_1 is not None else 100
            self.args.early_stop_mode = self.args.early_stop_mode_stage_1
            if self.args.early_stop_mode_stage_1 == 'step':
                self.args.early_stop_epoch = self.args.early_stop_epoch_stage_1
                self.args.patience_step = self.args.patience_step_stage_1
                self.args.early_stop_step_step = self.args.early_stop_step_step_stage_1
                self.args.early_stop_start_step = self.args.early_stop_start_step_stage_1
                
            elif self.args.early_stop_mode_stage_1 == 'epoch':
                self.args.patience = self.args.patience_stage_1
            else:
                raise ValueError(self.args.early_stop_mode_stage_1)
        elif self.args.training_stage == '2':
            self.lambda_l2 = self.args.lambda_l2_stage_2
            self.n_epoch = self.args.epoch_stage_2 if self.args.epoch_stage_2 is not None else 100
            self.args.early_stop_mode = self.args.early_stop_mode_stage_2
            if self.args.early_stop_mode_stage_2 == 'step':
                self.args.early_stop_epoch = self.args.early_stop_epoch_stage_2
                self.args.patience_step = self.args.patience_step_stage_2
                self.args.early_stop_step_step = self.args.early_stop_step_step_stage_2
                self.args.early_stop_start_step = self.args.early_stop_start_step_stage_2
            elif self.args.early_stop_mode_stage_2 == 'epoch':
                self.args.patience = self.args.patience_stage_2
            else:
                raise ValueError(self.args.early_stop_mode_stage_2)
        else:
            raise ValueError(self.args.training_stage)
        logging.info(f"training stage {self.args.training_stage}, lambda_l2: {self.lambda_l2}")
        logging.info(f"training stage {self.args.training_stage}, n_epoch: {self.n_epoch}")

    def evaluate(self, data_loader, current_epoch):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc=f"validation", smoothing=0, mininterval=1.0)
            for i, batch_data_pack in enumerate(tk0):
                x, y, idx = batch_data_pack
                x = x.to(self.device)
                # x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = self.model(x, current_epoch, current_step=i) # current_epoch=None means not in training mode
                targets.extend(y.tolist())
                if self.args.model == 'finalnet':
                    predicts.extend(y_pred["y_pred"].tolist())
                else:
                    predicts.extend(y_pred.tolist())
        return roc_auc_score(targets, predicts)
    
    def test(self, data_loader, evaluate_fns):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="test", smoothing=0, mininterval=1.0)
            start_time = dt.datetime.now()
            for i, batch_data_pack in enumerate(tk0):
                x, y, idx = batch_data_pack
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x, current_epoch=None, current_step=i)
                targets.extend(y.tolist())
                if self.args.model == 'finalnet':
                    predicts.extend(y_pred["y_pred"].tolist())
                else:
                    predicts.extend(y_pred.tolist())
            end_time = dt.datetime.now()
            logging.info('infer time: {} s'.format((end_time - start_time).total_seconds()))
        for evaluate_fn in evaluate_fns:
            if evaluate_fn == 'auc':
                auc = roc_auc_score(targets, predicts)
                logging.info(f'test auc: {auc}')
            elif evaluate_fn == 'logloss':
                logloss = log_loss(targets, predicts)
                logging.info(f'test logloss: {logloss}')
        return auc, logloss
    

        
