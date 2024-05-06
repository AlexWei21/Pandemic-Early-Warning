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