from run_training.train_past_guided import run_training
from datetime import datetime

target_training_len = 56
pred_len = 84

run_training(### Training Args
            lr = 1e-5,
            batch_size = 1024,
            target_training_len = target_training_len, 
            pred_len = pred_len, 
            record_run = True,
            max_epochs = 10000,
            log_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/logs/',
            ### Model Args
            loss = 'Combined_Loss',
            dropout=0.0,
            past_pandemics=['dengue','ebola','sars','2010-2017_influenza'],
            target_self_tuning=True,
            include_death=False,
            population_weighting= False,
            selftune_weight=1,
            use_lr_scheduler=True,
            loss_mae_weight = 0.5,
            loss_mape_weight = 100,
            output_dir=f"/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/{datetime.today().strftime('%m-%d-%H00')}_{target_training_len}-{pred_len}/",)