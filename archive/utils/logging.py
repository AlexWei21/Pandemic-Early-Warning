
def delphi_parameter_wise_loss_logging(loss_list, state):

    return {
        f'{state}_alpha': loss_list[0],
        f'{state}_days': loss_list[1],
        f'{state}_r_s': loss_list[2],
        f'{state}_r_dth': loss_list[3],
        f'{state}_p_dth': loss_list[4],
        f'{state}_r_dthdecay': loss_list[5],
        f'{state}_k1': loss_list[6],
        f'{state}_k2': loss_list[7],
        f'{state}_jump': loss_list[8],
        f'{state}_t_jump': loss_list[9],
        f'{state}_std_normal': loss_list[10],
        f'{state}_k3': loss_list[11],
    }