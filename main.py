import os
import random
import numpy as np
import pandas as pd

import torch
from models.deepifsac import DeepIFSAC
from utils import imputed_data
from data_loader import ( 
                         pre_process_deepifsac, 
                         generate_data_loader
                        )

TRAIN_EPOCHS = 10
BATCHSIZE = 128
EPOCHS = 1
MISSING_TYPE = 'mcar'
MISSING_RATE = 0.5
FLAG_TRAIN = 1
DATASET = "COM_PRODUTO"
DS_SEED = 0

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

    X_train = pd.read_csv("X_dataset_11_balance_missing20.csv").astype(np.float32)
    con_columns = X_train.columns
    con_idxs = list(range(len(X_train.columns)))

    temp = X_train.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    mean_features = X_train.loc[:, con_columns].mean().values.astype(np.float32)
    std_features = X_train.loc[:, con_columns].std().values.astype(np.float32)
    median_features = X_train.loc[:, con_columns].median().values.astype(np.float32)
    
    for i, col in enumerate(X_train.columns.values[con_idxs]):
         X_train.loc[:, col] = X_train.loc[:, col].fillna(mean_features[con_idxs[i]])
    
    _, t_mask = imputed_data(X_train.values, corruptor_settings)
    t_mask = np.array(t_mask.cpu()).copy().astype(np.int64) 


    ## STARTS HERE  
    X_train, imp_X_train, cat_dims = pre_process_deepifsac(X_train, t_mask, mean_features, median_features, con_idxs)
    train_loader = generate_data_loader(X_train, imp_X_train, t_mask, mean_features, std_features, cat_cols=[], create_ds=True, X_mask=nan_mask.values, shuffle_data=True)
   

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

    all_predictions_train, nrmse_train = model.transform(train_loader, device)
    print('NRMSE for continuous features on the train set:', nrmse_train.item())
    pd.DataFrame(all_predictions_train).to_csv("imputed_train_set.csv")

    # all_predictions_val, nrmse_val = model.transform(val_loader, device)
    # print('NRMSE for continuous features on the validation set:', nrmse_val.item())
    # pd.DataFrame(all_predictions_val).to_csv("imputed_val_set.csv")

if __name__ == '__main__':
    main()