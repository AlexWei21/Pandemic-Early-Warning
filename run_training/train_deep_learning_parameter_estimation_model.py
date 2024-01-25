from model.deep_learning_model_for_parameter_estimation import naive_nn
from run_training.training_utils import get_weight_estimation_model, run_compartment_model
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

def run_weight_estimation_model(past_pandemic_list:list,
                                target_pandemic:str,
                                num_of_compartment_edges:int,
                                num_of_trainable_parameters:int,
                                num_hidden:int = 2,
                                hidden_dim: int = 512,
                                compartment_model:str = 'SEIRD',
                                weight_estimation_model:str = 'Naive_nn',
                                n_epochs = 100,
                                batch_size = 10,
                                target_training_len = 30,
                                lr = 1e-4,
                                pred_len = 60,
                                ):
    
    past_pandemic_data = []

    for pandemic in past_pandemic_list:
        if pandemic == 'Dengue':
            past_pandemic_data.extend(process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_dengue_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'Ebola':
            past_pandemic_data.extend(process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_ebola_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'Influenza':
            past_pandemic_data.extend(process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_influenza_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'MPox':
            past_pandemic_data.extend(process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_mpox_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'SARS':
            past_pandemic_data.extend(process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_sars_data_objects.pickle',
                                                   raw_data=False))
        else:
            print(f"{pandemic} not in the processed data list, please process the data prefore running the model, skipping {[pandemic]}")       
    
    target_pandemic_data = process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)
    
    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=past_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,)

    train_data_loader = DataLoader(past_pandemic_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=past_pandemic_dataset.collate_fn,
                                   drop_last=True)

    # a = past_pandemic_dataset.__getitem__(1)
    # b = past_pandemic_dataset.__getitem__(2)

    # x = past_pandemic_dataset.collate_fn([a,b])

    # print(x)

    # print(next(iter(train_data_loader)))

    num_meta_data = len(target_pandemic_data[0].pandemic_meta_data)

    model_input_dim = target_training_len + num_meta_data
    model_output_dim = num_of_trainable_parameters + num_of_compartment_edges
    
    weight_estimation_model = get_weight_estimation_model(model_name=weight_estimation_model,
                                                          input_dim=model_input_dim,
                                                          output_dim=model_output_dim,
                                                          num_hidden=num_hidden,
                                                          hidden_dim=hidden_dim)
    
    optimizer = optim.Adam(weight_estimation_model.parameters(),
                           lr=lr)
    
    run = wandb.init()

    print (">>>>> Model Infrastructure")
    print(weight_estimation_model)

    for epoch in tqdm(range(n_epochs)):

        terrible_sample_list = []

        for i,data in enumerate(tqdm(train_data_loader, leave= False),0):

            optimizer.zero_grad()
            
            output = weight_estimation_model(data['model_input'])    

            edge_weights = output[:,:num_of_compartment_edges]
            params = output[:,num_of_compartment_edges:]

            loss, terrible_samples = run_compartment_model(data = data,
                                        predicted_edge_weights=edge_weights,
                                        predicted_parameters=params,
                                        target_training_len = target_training_len,
                                        pred_len=pred_len,
                                        # compartment_model = compartment_model,
                                        batch_size=batch_size,)

            run.log({"MAPE_Loss":loss, "epoch": epoch})

            loss.backward()

            optimizer.step()

            terrible_sample_list.append(terrible_samples)

        for item in terrible_sample_list:
            print(item)

run_weight_estimation_model(past_pandemic_list = ['Dengue','Ebola','Influenza','MPox','SARS'],
                            target_pandemic = 'Covid_19',
                            num_of_compartment_edges = 11,
                            num_of_trainable_parameters = 12,
                            num_hidden = 2,
                            hidden_dim = 512,
                            compartment_model = 'DELPHI',
                            weight_estimation_model = 'Naive_nn',
                            n_epochs = 100,
                            batch_size = 64,
                            target_training_len = 30,
                            lr = 0.05)