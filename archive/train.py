from run_training.train_deep_learning_parameter_estimation_model import run_weight_estimation_model

run_weight_estimation_model(past_pandemic_list = ['Dengue','Ebola','Influenza','MPox','SARS'],
                            root_dir = [''],
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