from run_training.train_naive_compartment_model import run_compartment

if __name__ == '__main__':
    run_compartment(look_back_len = 100,
                    pred_len = None)