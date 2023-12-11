from model.Naive_Transformer import Naive_Transformer
from data.dataset import Pandemic_Dataset
from utils.utils import generate_square_subsequent_mask, adjust_learning_rate
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import time
import numpy as np
from torchmetrics.regression import MeanAbsolutePercentageError
from utils.scheduler import Scheduler

if __name__ == '__main__':

    root_dir = 'F:/Pandemic-Database/Processed_Data/'
    target_file_path = "Covid_19/Covid_World_Domain_Daily_CumCases.csv"
    past_file_path = ["Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv",
                      "Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv",
                      "Monkeypox/Mpox_World_Country_Daily_CumCases.csv",
                      "SARS/SARS_World_Country_Daily_CumCases.csv",
                      "Influenza/Influenza_World_Domain_Weekly_CumCases.csv"]
    target_pandemic_name = 'Covid'
    past_pandemic_name = ['Dengue',
                          'Ebola',
                          'MPox',
                          'SARS',
                          'Influenza']
    target_data_frequency = 'Daily'
    past_data_frequency=['Weekly',
                         'Weekly',
                         'Daily',
                         'Daily',
                         'Weekly']
    
    look_back_len = 30
    pred_len = 60
    data_smoothing=True
    raw_data = True
    data_smoothing=True

    lr = 0.0001

    num_epoch = 100

    train_dataloader = Pandemic_Dataset(root_dir=root_dir,
                                        target_file_path=target_file_path,
                                        past_file_path=past_file_path,
                                        target_pandemic_name=target_pandemic_name,
                                        past_pandemic_name=past_pandemic_name,
                                        raw_data=raw_data,
                                        flag = 'train',
                                        look_back_len=look_back_len,
                                        pred_len=pred_len,
                                        data_smoothing=data_smoothing,
                                        target_data_frequency=target_data_frequency,
                                        past_data_frequency=past_data_frequency)
    
    test_dataloader = Pandemic_Dataset(root_dir=root_dir,
                                        target_file_path=target_file_path,
                                        past_file_path=past_file_path,
                                        target_pandemic_name=target_pandemic_name,
                                        past_pandemic_name=past_pandemic_name,
                                        raw_data=raw_data,
                                        flag = 'test',
                                        look_back_len=look_back_len,
                                        pred_len=pred_len,
                                        data_smoothing=data_smoothing,
                                        target_data_frequency=target_data_frequency,
                                        past_data_frequency=past_data_frequency)
    
    model = Naive_Transformer(len_look_back=look_back_len,
                              len_pred=pred_len,
                              batch_first=True)
    
    model_optim = optim.Adam(model.parameters(), lr=lr)
    criterion = MeanAbsolutePercentageError()

    scheduler = Scheduler(optimizer=model_optim,
                          dim_embed=512,
                          warmup_steps=100)

    tgt_mask = generate_square_subsequent_mask( 
        dim1=60,
        dim2=60
        )

    training_data = DataLoader(train_dataloader,32, collate_fn=train_dataloader.collate_fn)

    for epoch in range(num_epoch):
        iter_count = 0
        train_loss = []
        model.train() 
        # epoch_time = time.time()

        for i, batch in enumerate(training_data):
            iter_count += 1
            model_optim.zero_grad()

            x = batch['x'].unsqueeze(2)
            y = batch['y'].unsqueeze(2)
            meta_data = batch['meta_data'].unsqueeze(2)
            decoder_input = batch['decoder_input'].unsqueeze(2)

            output = model(src=x, 
                           tgt=decoder_input, 
                           meta_data=meta_data,
                           tgt_mask=tgt_mask)

            loss = criterion(output,y)

            print(loss.item())

            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((num_epoch - epoch) * len(train_dataloader) - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            loss.backward()
            model_optim.step()

            scheduler.step()
        
        # adjust_learning_rate(model_optim, epoch+1, lradj='type2', learning_rate=None) ## Use a set of fixed values of lr

    print(train_loss)