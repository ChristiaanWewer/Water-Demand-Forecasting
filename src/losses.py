import torch
import torch.nn as nn


class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(MultiQuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

    def forward(self, y, y_hat, masked_loss=True):

        # y dimension: (batch_size, horizon_length, 1)
        # y_hat dimension: (batch_size, horizon_length, quantiles)
        
        # dimension of y_pred and y_true are: (batch_size, horizon_length, quantiles)
        losses = 0
        if masked_loss:
            nan_mask = ~torch.isnan(y)
        else:
            nan_mask = torch.ones_like(y).bool()
                
        for i, q in enumerate(self.quantiles):

            y_quantile = y_hat[:, :, i].unsqueeze(-1)

            
            diff = y-y_quantile
            mask = diff > 0
            loss = torch.mean((q * diff * mask - (1-q) * diff * ~mask)[nan_mask])


            losses += loss
        losses /= self.num_quantiles

        return losses
