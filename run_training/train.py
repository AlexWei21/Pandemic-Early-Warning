from model.Naive_Transformer import Naive_Transformer
from data.dataset import Pandemic_Dataset
from utils.utils import generate_square_subsequent_mask
from torch.utils.data import DataLoader

def _run_model(model,x,y,decoder_input,src_mask=None, target_mask=None):

    output = model(src=x,
                   tgt=decoder_input,
                   target_mask=target_mask)
    
    

if __name__ == '__main__':
    train_dataloader = Pandemic_Dataset(root_dir = 'F:/Pandemic-Database/Processed_Data/', 
                         target_file_path = "Covid_19/Covid_World_Domain_Daily_CumCases.csv", 
                         past_file_path = ["Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv",
                                           "Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv",
                                           "Monkeypox/Mpox_World_Country_Daily_CumCases.csv",
                                           "SARS/SARS_World_Country_Daily_CumCases.csv",
                                           "Influenza/Influenza_World_Domain_Weekly_CumCases.csv"],
                         target_pandemic_name = 'Covid',
                         past_pandemic_name = ['Dengue',
                                               'Ebola',
                                               'MPox',
                                               'SARS',
                                               'Influenza'],
                         raw_data=True,
                         flag = 'train',
                         look_back_len=30,
                         pred_len=60,
                         data_smoothing=True,
                         target_data_frequency='Daily',
                         past_data_frequency=['Weekly',
                                              'Weekly',
                                              'Daily',
                                              'Daily',
                                              'Weekly'])
    
    model = Naive_Transformer(len_look_back=30,
                              len_pred=60,
                              batch_first = True)
    
    tgt_mask = generate_square_subsequent_mask( 
        dim1=60,
        dim2=60
        )

    training_data = DataLoader(train_dataloader,32, collate_fn=train_dataloader.collate_fn)

    i,batch = next(enumerate(training_data))

    x = batch['x'].unsqueeze(2)
    y = batch['y'].unsqueeze(2)
    meta_data = batch['meta_data'].unsqueeze(2)
    decoder_input = batch['decoder_input'].unsqueeze(2)

    # print(x.shape)
    # print(decoder_input.shape)

    # print(model.input_projection.weight.dtype)

    output = model(src=x,
                   tgt=decoder_input,
                   meta_data=meta_data,
                   tgt_mask=tgt_mask)
    
    print(output)

