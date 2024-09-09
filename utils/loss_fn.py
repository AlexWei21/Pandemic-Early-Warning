import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

class Weighted_L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):

        mae = nn.L1Loss()


class WMAE(nn.Module):
    def __init__(self, output_weights):
        super().__init__()
        self.output_weights = output_weights
    
    def forward(self, y_pred, y_true):

        mae = nn.L1Loss(reduction='none')

        loss = mae(y_pred, y_true)
        loss = loss * torch.tensor(self.output_weights).to(loss)
        loss = torch.mean(loss, dim = 0)
        
        unscaled_loss = loss / torch.tensor(self.output_weights).to(loss)
        
        loss = torch.mean(loss)

        return loss, unscaled_loss

class MAPE(nn.Module):
    def __init__ (self, reduction = 'mean'):
        super(MAPE, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        
        if self.reduction == 'mean':
            return MeanAbsolutePercentageError()(y_pred,y_true)
        elif self.reduction == 'none':
            return torch.abs(y_true - y_pred) / y_true
        else:
            raise NotImplementedError

class Combined_Loss(nn.Module):
    def __init__ (self, reduction = 'mean', mae_weight = 1.0, mape_weight = 1.0):
        super(Combined_Loss, self).__init__()
        self.reduction = reduction
        self.mae_weight = mae_weight
        self.mape_weight = mape_weight
    
    def forward(self, y_pred, y_true):

        if self.reduction == 'mean':
            loss = self.mae_weight * nn.L1Loss(reduction='mean')(y_pred, y_true) \
                   + self.mape_weight * MAPE(reduction='mean')(y_pred, y_true)
        elif self.reduction == 'none':
            loss = self.mae_weight * nn.L1Loss(reduction='none')(y_pred, y_true) \
                   + self.mape_weight * MAPE(reduction='none')(y_pred, y_true)
        else:
            raise NotImplementedError
        
        return loss


class DMAPE(nn.Module):
    def __init__(self, delta_t):
        super(DMAPE, self).__init__()
        self.delta_t = delta_t

    def forward(self, y_pred, y_true, previous_info):
        
        if len(previous_info[0]) < self.delta_t:
            print("delta_t is larger than the train_input time series length.")
            exit()

        y_pred_base_data = torch.cat([previous_info, y_pred], dim=1)[:,(len(previous_info[0]) - self.delta_t): -self.delta_t].detach()
        y_true_base_data = torch.cat([previous_info, y_true], dim=1)[:,(len(previous_info[0]) - self.delta_t): -self.delta_t]

        diff_y_pred = y_pred - y_pred_base_data

        diff_y_pred = nn.ReLU()(diff_y_pred)

        diff_y_true = y_true - y_true_base_data

        mape = MeanAbsolutePercentageError().cuda() if torch.cuda.is_available() else MeanAbsolutePercentageError()

        loss = mape(diff_y_pred, diff_y_true)
        
        return loss