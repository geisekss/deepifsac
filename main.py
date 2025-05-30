import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from models.deepifsac import DeepIFSAC
from data_loader import data_prep_dataFrame, Dataset_imputed

TRAIN_EPOCHS = 10
BATCHSIZE = 128
EPOCHS = 1
MISSING_TYPE = 'mcar'
MISSING_RATE = 0.5
FLAG_TRAIN = 1
DATASET = "BALANCE_SCALE"
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

    torch.manual_seed(0)
    np.random.seed(1)
    
    df = pd.read_csv("X_dataset_11_balance.csv")
    cat_dims, cat_idxs, con_idxs, X_train, X_test, train_mean, train_std = data_prep_dataFrame(df,seed=DS_SEED)
    train_mean_std = np.array([train_mean, train_std]).astype(np.float32)
    
    # In case X has only numerical data: cat_idxs = []
    train_ds = Dataset_imputed(X_train, continuous_mean_std=train_mean_std, cat_cols=cat_idxs)
    train_ds.generate_imputed(corruptor_settings)
    train_imp_mean_std = (train_ds.imp_mean, train_ds.imp_std)
    test_ds = Dataset_imputed(X_test, continuous_mean_std=train_mean_std, cat_cols=cat_idxs)
    test_ds.generate_imputed(corruptor_settings, imp_mean_std=train_imp_mean_std)

    trainloader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, num_workers=0)

    # Append 1 for CLS token.
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

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
        model.fit(trainloader, filename_metrics, TRAIN_EPOCHS, fold_key, device)

        # Save pretrained model.
        folder = './results/model_weights'
        if(not os.path.exists(folder)):
            os.makedirs(folder)

        torch.save(model.state_dict(), MODEL_PATH)
        print("DeepIFSAC TRAINING DONE")

    else:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    all_predictions_train, nrmse_train = model.transform(trainloader, device, train_mean_std, train_imp_mean_std)
    print('NRMSE for continuous features on the train set:', nrmse_train.item())
    pd.DataFrame(all_predictions_train).to_csv("results/imputed_train_set.csv")

    all_predictions_test, nrmse_teste = model.transform(testloader, device, train_mean_std, train_imp_mean_std)
    print('NRMSE for continuous features on the test set:', nrmse_teste.item())
    pd.DataFrame(all_predictions_test).to_csv("results/imputed_test_set.csv")


if __name__ == '__main__':
    main()