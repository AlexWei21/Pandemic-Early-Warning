from model.deep_learning_model_for_parameter_estimation import naive_nn

def get_weight_estimation_model(model_name,
                                input_dim,
                                output_dim,
                                num_hidden = 2,
                                hidden_dim = 512,
                                pred_len = 60,
                                target_training_len = 30,
                                dnn_output_range = None,
                                output_dir = None,
                                device = 'cpu',
                                batch_size = 64,
                                population_normalization = True,
                                dropout = 0.0,
                                predict_parameters_only = False,):

    assert (model_name in ['Naive_nn']), "Provided weight estimation model is not supported"

    if model_name == 'Naive_nn':
        return naive_nn(input_dim=input_dim,
                        output_dim=output_dim,
                        num_hidden=num_hidden,
                        hidden_dim=hidden_dim,
                        pred_len = pred_len,
                        target_training_len=target_training_len,
                        dnn_output_range=dnn_output_range,
                        output_dir = output_dir,
                        device = device,
                        batch_size=batch_size,
                        population_normalization = population_normalization,
                        dropout = dropout,
                        predict_parameters_only = predict_parameters_only,)
    else:
        return -1