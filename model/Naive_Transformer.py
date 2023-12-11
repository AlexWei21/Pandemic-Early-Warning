import torch
import torch.nn as nn
import lightning as L
import layers.positional_encoder as positional_encoder


class Naive_Transformer(L.LightningModule):
    def __init__(self,
                 len_look_back,
                 len_pred,
                 d_model=512,
                 dropout=0.2,
                 dropout_pos_enc = 0.1,
                 lr=1e-4,
                 batch_first = True):
        
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        self.positional_encoding_layer = positional_encoder.PositionalEncoder(
            d_model=d_model,
            dropout=dropout_pos_enc
        )

        self.input_projection = nn.Linear(1, d_model)

        self.meta_projection = nn.Linear(1,d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4*d_model,
            batch_first=batch_first
        )

        self.output_projection = nn.Linear(1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4*d_model,
            batch_first=batch_first
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.readout = nn.Linear(d_model,1)

    def forward(self, 
                src,
                tgt,
                meta_data,
                src_mask = None,
                tgt_mask = None):

        src = self.input_projection(src)
        meta_data = self.meta_projection(meta_data)

        src = self.positional_encoding_layer(src)

        src = torch.cat((src,meta_data),1)

        src = self.encoder(src)

        decoder_output = self.output_projection(tgt)


        decoder_output = self.decoder(
            tgt = decoder_output,
            memory = src,
            tgt_mask = tgt_mask,
            memory_mask = src_mask
        )

        decoder_output = self.readout(decoder_output)

        return decoder_output
    
if __name__ == '__main__':

    model = Naive_Transformer(100,200)

    print(model)