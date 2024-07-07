from model.deep_learning_model_for_parameter_estimation import naive_nn
from utils.training_utils import run_compartment_model
from utils.get_models import get_weight_estimation_model
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torchmetrics.regression import MeanAbsolutePercentageError
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.loss_fn import DMAPE
from pathlib import Path
from matplotlib import pyplot as plt

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
                                record_run = False,
                                plot_result = False,
                                output_dir = None,
                                loss_fn = 'MAPE',
                                delta_t = 30,
                                opt = 'Adam',
                                seed = 15,
                                log_parameter = False,
                                population_normalization = True,
                                dropout = 0.0,
                                predict_parameters_only: bool = False,
                                perfect_parameters_dir: str = "/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_params_covid.csv",
                                ):
    
    torch.manual_seed(seed)

    config = dict(past_pandemic_list = past_pandemic_list,
                  target_pandemic = target_pandemic,
                  num_of_compartment_edges = num_of_compartment_edges,
                  num_of_trainable_parameters = num_of_trainable_parameters,
                  num_hidden = num_hidden,
                  hidden_dim = hidden_dim,
                  compartment_model = compartment_model,
                  weight_estimation_model = weight_estimation_model,
                  n_epochs = n_epochs,
                  batch_size = batch_size,
                  target_training_len = target_training_len,
                  lr = lr,
                  pred_len = pred_len,
                  loss_fn = loss_fn,
                  delta_t = delta_t,
                  opt = opt,
                  population_normalization = population_normalization)
    
    ### Data Loading
    past_pandemic_data = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for pandemic in past_pandemic_list:
        if pandemic == 'Dengue':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_dengue_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'Ebola':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_ebola_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'Influenza':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_influenza_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'MPox':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_mpox_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'SARS':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_sars_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'Covid_19':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_covid_data_objects.pickle',
                                                   raw_data=False))
        else:
            print(f"{pandemic} not in the processed data list, please process the data prefore running the model, skipping {[pandemic]}")       
    
    target_pandemic_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)
    
    ### Debug: Toy Data
    # toy_data = [item for item in past_pandemic_data if (item.domain_name == 'Massachusetts')]
    # toy_data = [item for item in past_pandemic_data if (item.domain_name == 'Washington') | (item.domain_name == 'Massachusetts') | (item.domain_name == 'Ohio')| (item.domain_name == 'North Carolina')]
    toy_data = [item for item in past_pandemic_data if (item.domain_name == 'Ohio') | (item.domain_name == 'Massachusetts')]
    past_pandemic_data = toy_data
    # print(len(toy_data))

    ### Data Loaders
    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=past_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False)

    train_data_loader = DataLoader(past_pandemic_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=past_pandemic_dataset.collate_fn,
                                   drop_last=False,
                                   )

    ## Dimension Definition
    num_meta_data = len(target_pandemic_data[0].pandemic_meta_data)
    model_input_dim = target_training_len + num_meta_data
    model_output_dim = num_of_trainable_parameters + num_of_compartment_edges
    
    ## Define Range of Output
    output_max = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,
                               1.0,       # Alpha 0.7
                               2,       # days 1.4
                               1.0,         # r_s 1
                               1.0,      # r_dth 0.96
                               0.32,      # p_dth 0.32
                               4.5,       # r_dthdecay
                               0.2 ,      # k1
                               454.0,     # k2
                               8.22,      # jump
                               209.0,     # t_jump
                               2.0,       # std_normal
                               0.1        # k3
                               ])

    ## Model Definition
    weight_estimation_model = get_weight_estimation_model(model_name=weight_estimation_model,
                                                          input_dim=model_input_dim,
                                                          output_dim=model_output_dim,
                                                          num_hidden=num_hidden,
                                                          hidden_dim=hidden_dim,
                                                          pred_len = pred_len,
                                                          target_training_len = target_training_len,
                                                          dnn_output_range = output_max,
                                                          output_dir = output_dir,
                                                          device = device,
                                                          batch_size=batch_size,
                                                          population_normalization = population_normalization,
                                                          dropout = dropout,
                                                          predict_parameters_only = predict_parameters_only
                                                          ).to(device)

    if opt == 'Adam':
        optimizer = optim.Adam(weight_estimation_model.parameters(),
                            lr=lr)
    elif opt == 'AdamW':
        optimizer = optim.AdamW(weight_estimation_model.parameters(),
                            lr=lr)
    elif opt == 'SGD':
        optimizer = optim.SGD(weight_estimation_model.parameters(),
                            lr=lr)
    elif opt == 'Adagrad':
        optimizer = optim.Adagrad(weight_estimation_model.parameters(),
                                  lr = lr)
    elif opt == 'RMSprop':
        optimizer = optim.RMSprop(weight_estimation_model.parameters(),
                                  lr = lr)
        
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode = 'min',
    #                                                        factor = 0.1,
    #                                                        patience = 10,
    #                                                        threshold = 0.0001)

    if loss_fn == 'MAPE' or loss_fn == 'max_MAPE' or loss_fn == 'case_weighted_MAPE':
        criterion = MeanAbsolutePercentageError().cuda() if torch.cuda.is_available() else MeanAbsolutePercentageError()
    elif loss_fn == 'MAE' or loss_fn == 'max_MAE' or loss_fn == 'case_weighted_MAE':
        criterion = torch.nn.L1Loss().cuda() if torch.cuda.is_available() else torch.nn.L1Loss()
    elif loss_fn == 'DMAPE':
        criterion = DMAPE(delta_t=delta_t).cuda() if torch.cuda.is_available() else DMAPE(delta_t=30)
    
    if record_run:
        run = wandb.init(project = "Hospitalization_Prediction",
                         config = config)
        

    print (">>>>> Model Infrastructure")
    print(weight_estimation_model)
    print(config)

    pandemic_name_list = []
    country_list = []
    domain_list = []
    true_case_list = []
    output_list = []
    dnn_output_list = []

    sample_dist_list = []
    sample_dist_list_before_scaling = []
    sample_dist_list_dnn_output = []

    mass_output = []
    ohio_output = []

    last_layer_weights = []

    # Custom loss weight experiment
    # weight_dict = {"Massachusetts": 0.78,
    #                "Ohio": 0.22 }

    ### Training Loop
    for epoch in tqdm(range(n_epochs)):

        terrible_sample_list = []   
        performance_df = []
        avg_loss = []
        ts_input_list = []
        dnn_output_before_scaling_list = []
        dnn_output_debug_list = []

        tspan = np.arange(0, target_training_len + pred_len)

        for i,data in enumerate(tqdm(train_data_loader, leave= False),0):
            
            optimizer.zero_grad()

            output, dnn_output, ts_input, dnn_output_before_scaling = weight_estimation_model(data, 
                                                            sigmoid_scale = 0.1)

            print(data['country_name'], data['domain_name'])

            if data['domain_name'][0] == 'Massachusetts':
                mass_output.append(dnn_output[:,11:].tolist()[0])
            elif data['domain_name'][0] == 'Ohio':
                ohio_output.append(dnn_output[:,11:].tolist()[0])
            
            if epoch == n_epochs - 1:
                pandemic_name_list.append(data['pandemic_name'])
                country_list.append(data['country_name'])
                domain_list.append(data['domain_name'])
                true_case_list.append(data['cumulative_case_number'])
                output_list.append(output)
                dnn_output_list.append(dnn_output)

            pred_case = output[:, target_training_len:,15]

            true_case_data = []
            train_case_data = []

            for item in data['cumulative_case_number']:
                true_case_data.append(item[target_training_len:target_training_len+pred_len])
                train_case_data.append(item[:target_training_len])

            y_true = torch.tensor(np.array(true_case_data)).to(device)

            if loss_fn == 'DMAPE':
                previous_info = data['model_input'][:,:target_training_len].to(device)
                loss = criterion(pred_case, y_true, previous_info)

            elif loss_fn.startswith('max'):
                loss = []
                for i in range(y_true.shape[0]):
                    loss.append(criterion(pred_case[i,:], y_true[i,:]))
                loss = torch.stack(loss)

                max_idx = torch.argmax(loss)
                max_loss_mask = torch.zeros(batch_size, dtype=float).to(device)
                max_loss_mask[max_idx] = 1

                loss = torch.matmul(loss,max_loss_mask)

            elif loss_fn.startswith('case_weighted'):
                train_case_data = torch.tensor(np.array(train_case_data))

                loss = []
                for i in range(y_true.shape[0]):
                    loss.append(criterion(pred_case[i,:], y_true[i,:]))
                loss = torch.stack(loss)

                max_case_mask = torch.max(train_case_data, dim = 1).values
                max_case_mask = torch.div(max_case_mask, torch.max(max_case_mask)).to(loss.dtype).to(device)

                loss = torch.matmul(loss, max_case_mask)
                
            else:        
                loss = criterion(pred_case, y_true)

            ts_input_list.append(torch.flatten(ts_input))
            dnn_output_debug_list.append(torch.flatten(dnn_output))
            dnn_output_before_scaling_list.append(dnn_output_before_scaling)
            avg_loss.append(loss.item())

            print(loss.item())

            # Custom Loss Weight 
            # loss_weight = weight_dict[data['domain_name'][0]]
            # loss = loss * loss_weight
            # print(loss_weight)
            # print(loss.item())

            loss.backward()    

            ## Debug: Check if there's None Gradient
            for name, param in weight_estimation_model.named_parameters():
                if name == 'readout.38.weight':
                    last_layer_weights.append(torch.flatten(param).tolist())
                if param.grad is None:
                    print(name, param.grad)

            if record_run:
                run.log({"Step_Loss":loss.item(), "epoch": epoch})

            optimizer.step()

        # scheduler.step(np.mean(avg_loss))
        # print(scheduler.optimizer.param_groups[0]['lr'])
        print(np.mean(avg_loss))   

        # Sample distance experiemnt           
        ts_input_distance = nn.PairwiseDistance(p=2)(ts_input_list[0], ts_input_list[1])
        dnn_output_before_scaling_distance = nn.PairwiseDistance(p=2)(dnn_output_before_scaling_list[0],dnn_output_before_scaling_list[1])
        dnn_output_distance = nn.PairwiseDistance(p=2)(dnn_output_debug_list[0],dnn_output_debug_list[1])
        sample_dist_list.append(ts_input_distance.item())
        sample_dist_list_before_scaling.append(dnn_output_before_scaling_distance.item())
        sample_dist_list_dnn_output.append(dnn_output_distance.item())
        print("l2 Distance between inputs: ", ts_input_distance.item())
        print("l2 Distance between DNN output before scaling is: ", dnn_output_before_scaling_distance.item())
        print("l2 Distance between DNN output is: ", dnn_output_distance.item())

        if record_run:
            run.log({"Epoch_Loss":loss.item(), "epoch": epoch})
    
    # Sample distance experiment
    x = np.arange(0,n_epochs)    
    plt.plot(x, sample_dist_list, label = 'Sample Distance after Time Series Encoding')
    plt.plot(x, sample_dist_list_before_scaling, label = 'Sample Distance after Feedforward layer before Sigmoid Scaling')
    plt.plot(x, sample_dist_list_dnn_output, label = 'Sample Distance when given to ode solver')
    plt.show()
    
    # Weight Change Experiment
    # with open(output_dir + 'parameter_change/last_layer_weight_change.txt', 'w') as f:
    #     for item in last_layer_weights:
    #         for i in item:
    #             f.write(str(i))
    #             f.write('    ')
    #         f.write('\n')

    # exit()

    # with open(output_dir + 'parameter_change/mass_weight_change.txt', 'w') as f:
    #     for item in mass_output:
    #         for i in item:
    #             f.write(str(i))
    #             f.write('    ')
    #         f.write('\n')

    # with open(output_dir + 'parameter_change/ohio_weight_change.txt', 'w') as f:
    #     for item in ohio_output:
    #         for i in item:
    #             f.write(str(i))
    #             f.write('    ')
    #         f.write('\n')    
    
    pandemic_name_list = [item for row in pandemic_name_list for item in row]
    country_list = [item for row in country_list for item in row]
    domain_list = [item for row in domain_list for item in row]
    true_case_list = [item for row in true_case_list for item in row]
    output_list = [item for row in output_list for item in row]
    dnn_output_list = [item for row in dnn_output_list for item in row]

        ### Stratified Result Examination
        # performance_df = pd.DataFrame(performance_df, columns=['Pandemic_Name','Loss'])

        # performance_df = (performance_df.groupby(['Pandemic_Name'])
        #               .agg([('Average_Loss','mean'),('Count', 'count')])
        #               .reset_index())

        # print(performance_df)

        # for item in terrible_sample_list:
        #     print(item)
    
    

    if log_parameter:

        for item in set(pandemic_name_list):
            Path(output_dir + f'{item}_parameter/').mkdir(parents=False,exist_ok=True)

        for i in range(len(dnn_output_list)):
            param_dict = {'alpha': dnn_output_list[i][11].item(),
                        'days': dnn_output_list[i][12].item(),
                        'r_s': dnn_output_list[i][13].item(),
                        'r_dth': dnn_output_list[i][14].item(),
                        'p_dth': dnn_output_list[i][15].item(),
                        'r_dthdecay': dnn_output_list[i][16].item(),
                        'k1': dnn_output_list[i][17].item(),
                        'k2': dnn_output_list[i][18].item(),
                        'jump': dnn_output_list[i][19].item(),
                        't_jump': dnn_output_list[i][20].item(),
                        'std_normal': dnn_output_list[i][21].item(),
                        'k3': dnn_output_list[i][22].item(),}
        
            with open(output_dir + f"{pandemic_name_list[i]}_parameter/{country_list[i]}_{domain_list[i]}__train{target_training_len}_test{pred_len}_parameter_logs.txt", "w") as f:
                for key in param_dict:
                    f.write(str(key))
                    f.write(": ")
                    f.write(str(param_dict[key]))
                    f.write('\n')
                f.write('\n')

    if plot_result:

        for item in set(pandemic_name_list):
            Path(output_dir + f'{item}_plot/').mkdir(parents=False,exist_ok=True)

        for i in range(len(output_list)): 
            fig = go.Figure()
            # print(output_list[i][:,15].tolist())
            fig.add_trace(go.Scatter(x=tspan, y=output_list[i][:,15].tolist(), mode = 'markers', name='Predicted Infections', line = dict(dash='dot')))
            fig.add_trace(go.Scatter(x=tspan, y=true_case_list[i].tolist(), mode = 'markers', name='Observed Infections', line = dict(dash='dot')))
            fig.add_trace(go.Scatter(x=tspan, y=output_list[i][:,1].tolist(), mode = 'markers', name='Predicted Exposed', line = dict(dash='dot')))

            fig.write_image(output_dir + f'{pandemic_name_list[i]}_plot/{country_list[i]}_{domain_list[i]}_train{target_training_len}_test{pred_len}_{loss_fn}.png')


run_weight_estimation_model(past_pandemic_list = ['Covid_19'],
                            target_pandemic = 'Covid_19',
                            num_of_compartment_edges = 11,
                            num_of_trainable_parameters = 12,
                            num_hidden = 20,
                            hidden_dim = 256,
                            compartment_model = 'DELPHI',
                            weight_estimation_model = 'Naive_nn',
                            n_epochs = 100,
                            batch_size = 1,
                            target_training_len = 30,
                            lr = 0.0001,
                            record_run = False,
                            pred_len = 60,
                            plot_result = True,
                            log_parameter = True,
                            output_dir = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/',
                            loss_fn='MAPE',
                            opt = 'AdamW',
                            seed = 15,
                            population_normalization = True,
                            dropout = 0.2,
                            predict_parameters_only = True,
                            perfect_parameters_dir = "/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_params_covid.csv", )