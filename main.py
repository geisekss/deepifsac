import os
import random
import numpy as np
import pandas as pd

import torch
from models.deepifsac import DeepIFSAC
from models.diffputter import DiffPuter
from utils import   (
                        save_results
                    )
from data_loader import ( 
                         load_impute_X,
                         pre_process_deepifsac, 
                         pre_process_diffputter,
                         generate_data_loader
                        )

TRAIN_EPOCHS = 10
BATCHSIZE = 128
EPOCHS = 1
MISSING_TYPE = 'mcar'
MISSING_RATE = 0.5
FLAG_TRAIN = 1
DATASET = "BALANCE"
DS_SEED = 0
N_TRIALS = 10
N_STEPS = 50
MAX_ITER = 10

def main():

    corruptor_settings = {
        'method': 'mcar_missing',
        'corruption_rate': 0.6,
        'missing': MISSING_RATE,
        'missing_type': MISSING_TYPE,
        'mice': 'LinearRegression'
    }

    device = torch.device(f"cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f"Device is {device}.")

    torch.manual_seed(1)
    np.random.seed(1)


    ## DIFFPUTTER    
    '''
    X_train, nan_mask, t_mask, mean_features, std_features, median_features, con_idxs = load_impute_X("housing.csv", corruptor_settings)
    X_train, X_train_miss, mask = pre_process_diffputter(X_train.values, t_mask, mean_features, std_features)
    X_train_miss_init = X_train_miss.copy()

    result_save_path = f'results/{DATASET}/{N_TRIALS}_{N_STEPS}'
    os.makedirs(result_save_path) if not os.path.exists(result_save_path) else None
    MODEL_PATH = f'./results/model_weights/diffputter'
    
    diffputer = DiffPuter(result_save_path = result_save_path,
                          num_trials = N_TRIALS, 
                          epochs_m_step = TRAIN_EPOCHS, 
                          patience_m_step = 300, 
                          hid_dim = 1024, 
                          device = device, 
                          lr = 1e-4, 
                          num_steps = N_STEPS, 
                          ckpt_dir = MODEL_PATH,
                          in_dim=X_train.shape[1])
    

    for iteration in range(MAX_ITER):
        print("Iteration:", iteration)
        os.makedirs(f'{MODEL_PATH}/{iteration}', exist_ok=True) if not os.path.exists(f'{MODEL_PATH}/{iteration}') else None

        
        if iteration > 0:
            print(f'Loading X_train_miss')
            X_train_miss = np.load(f'{MODEL_PATH}/{iteration}/Xmiss_iter{iteration}.npy') / 2

        train_loader = generate_data_loader(X_train_miss, nan_mask=t_mask, mean=mean_features, std=std_features, cat_cols=[])
       
        diffputer.fit(iteration, train_loader)
        rec_X = diffputer.transform(iteration, X_train, mask, X_train_miss_init)

        mae_train, rmse_train= diffputer.get_eval(rec_X, X_train, mask)
        print('in-sample', mae_train, rmse_train)

        save_results(result_save_path, {
                                            f'iteration{iteration}': 
                                                {
                                                    'MAE': mae_train, 
                                                    'RMSE': rmse_train}
                                                }
                    )
        diffputer.model.load_state_dict(torch.load(f'{MODEL_PATH}/{iteration}/model.pt'))
    ''' 
    ## DEEPIFSAC  
    
    X_train, nan_mask, t_mask, mean_features, std_features, median_features, con_idxs = load_impute_X("X_dataset_11_balance_missing20.csv", corruptor_settings)
    col_names = X_train.columns.to_list()
    X_train, imp_X_train, cat_dims = pre_process_deepifsac(X_train, t_mask, mean_features, std_features, median_features, con_idxs)
    train_loader = generate_data_loader(X_train, imp_X=imp_X_train, t_mask=t_mask, mean=mean_features, std=std_features, create_ds=True, X_mask=nan_mask.values, shuffle_data=True)
    

    cutmix_corruptor_settings = {
                            'method': 'cutmix',
                            'corruption_rate': 0.6,
                            'missing': MISSING_RATE,
                            'missing_type': MISSING_TYPE,
                            'mice': 'LinearRegression'
                        }
    
    model = DeepIFSAC(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        corruptor_settings=cutmix_corruptor_settings
    )
    
    MODEL_PATH = f'./results/model_weights/{DATASET}_{MISSING_TYPE}_{MISSING_RATE}_cutmix_model.pth'
    
    if FLAG_TRAIN:
        print('DeepIFSAC TRAINING')

        directory = './results/training_scores'
        if(not os.path.isdir(directory)):
            os.mkdir(directory)

        fold_key = f'fold_{DS_SEED}'
        filename_metrics = f'{directory}/train_{DATASET}_{MISSING_TYPE}_{MISSING_RATE}_cutmix.pkl' 
        model.fit(train_loader, filename_metrics, TRAIN_EPOCHS, fold_key, device)

        # Save pretrained model.
        folder = './results/model_weights'
        if(not os.path.exists(folder)):
            os.makedirs(folder)

        torch.save(model.state_dict(), MODEL_PATH)
        print("DeepIFSAC TRAINING DONE")

    else:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    mean_std = (train_loader.dataset.mean, train_loader.dataset.std)
    imp_mean_std = (train_loader.dataset.imp_mean, train_loader.dataset.imp_std)

    train_loader_pred = generate_data_loader(X_train, imp_X=imp_X_train, t_mask=t_mask, mean=mean_features, std=std_features, create_ds=True, X_mask=nan_mask.values, shuffle_data=False)
    all_predictions_train, nrmse_train = model.transform(train_loader_pred, mean_std, imp_mean_std, device)
    print('NRMSE for continuous features on the train set:', nrmse_train.item())
    df_preds = pd.DataFrame(all_predictions_train, columns=col_names)
    df_preds.to_csv("imputed_train_set.csv", index=None)

    # all_predictions_val, nrmse_val = model.transform(val_loader, device)
    # print('NRMSE for continuous features on the validation set:', nrmse_val.item())
    # pd.DataFrame(all_predictions_val).to_csv("imputed_val_set.csv")
    

if __name__ == '__main__':
    main()