import os
import json

import torch
import numpy as np

def get_nan_mask(X):
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    return nan_mask

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    
    
def recreate_empty_file(filename):
    """
    Checks if the specified file is empty, and if so, deletes it and creates a new empty file with the same name.

    Parameters:
    - filename (str): The path to the file to check and recreate if empty.
    """
    # Check if the file exists to avoid FileNotFoundError
    if os.path.exists(filename):
        # Check if the file is empty by looking at its size
        if os.path.getsize(filename) == 0:
            print(f"File {filename} is empty. Deleting and creating a new one.")
            os.remove(filename)  # Delete the empty file            
            # Create a new, empty file with the same name
            with open(filename, 'w') as f:
                pass  # 'pass' simply creates an empty block, resulting in an empty file
            print(f"New empty file {filename} created.")
        else:
            print(f"File {filename} is not empty.")
    else:
        # If the file doesn't exist, simply create a new one
        with open(filename, 'w') as f:
            pass  # Create a new empty file
        print(f"File {filename} did not exist and was created.")


def embed_data_mask(model, x_categ, x_cont, cat_mask, con_mask):
    device = x_cont.device
    # x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    x_cont_enc = torch.empty(n1,n2, model.dim)
    for i in range(model.num_continuous):
        x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)

    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_categ, x_categ_enc, x_cont_enc


def imputed_data(data, settings, opt = None):
    from corruptor import Corruptor
    if opt is not None:
        corruptor_settings ={
                    'method': 'draw',
                    'corruption_rate': 0.6,
                    'missing': opt.missing_rate,
                    'missing_type': 'mcar', #opt.missing_type
                    'mice': 'LinearRegression'
                }
        data = torch.tensor(data)
        corruptor= Corruptor(data, corruptor_settings)
        X_train_imp, mask = corruptor(data)
        X_train_imp = torch.tensor(X_train_imp)
        # print(X_train_imp)
    else:
        corruptor_x = Corruptor(data, settings)
        data, mask = corruptor_x(torch.tensor(data))
        # median = torch.nanmedian(data, dim=0).values
        # X_train_imp = torch.where(torch.isnan(data), median, data)

    return data, mask

def standardize_data(X: np.array, mean_X: np.array, std_X: np.array) -> np.array:

    X_stdized = (X - mean_X) / std_X / 2
    X_stdized = torch.tensor(X_stdized)
    return X_stdized

def z_score(X: np.array, mean_X: np.array, std_X: np.array) -> np.array:
    X_stdized = (X - mean_X) / std_X 
    X_stdized = torch.tensor(X_stdized)
    return X_stdized

def min_max_scaler(X: np.array, min_vals: np.array, max_vals: np.array) -> np.array:
    in_dim = X.shape[1]
    eps = 0.001

    for i in range(in_dim):

        X[:,i] = (X[:,i] - min_vals[i])/(max_vals[i] - min_vals[i] + eps)

    return X

def save_results(result_save_path, results):
    with open (f'{result_save_path}/result.txt', 'a+') as f:
        f.write(json.dumps(results))

    print(f'saving results to {result_save_path}')