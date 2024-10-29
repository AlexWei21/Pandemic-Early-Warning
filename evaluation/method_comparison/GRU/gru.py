import torch
import torch.nn as nn

class PF_GRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 pred_length: int,
                 sequence_length: int,
                 ):
        super(PF_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers,
                          batch_first = True,)
        self.fc = nn.Linear(hidden_size * sequence_length,
                             pred_length)
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size).to(x)
        
        out,_ = self.gru(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)

        return out

if __name__ == '__main__':

    input_size = 1
    batch_size = 64
    hidden_size = 128
    num_layers = 5
    pred_length = 28
    sequence_length = 56

    input = torch.randn(64,sequence_length,input_size)

    gru = PF_GRU(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 pred_length=pred_length,
                 sequence_length=sequence_length)
    
    output = gru(input)
    print(output.shape)