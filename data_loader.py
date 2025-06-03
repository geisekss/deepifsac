import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from utils import imputed_data, standardize_data

class Dataset_imputed(Dataset):
    def __init__(self, X, t_mask=None, imp_X=None, continuous_mean_std=None, imp_mean_std=None, cat_cols=[]):
        
        self.cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        self.X = X['data'].copy()
        self.con_cols = list(set(np.arange(self.X.shape[1])) - set(cat_cols))
        self.Xcat = self.X[:,self.cat_cols].copy().astype(np.int64) #categorical columns
        self.Xcont = self.X[:,self.con_cols].copy().astype(np.float32) #numerical columns
        self.Xcat_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.Xcont_mask = X_mask[:,self.con_cols].copy().astype(np.int64) #numerical columns

        self.cls = np.zeros((self.X.shape[0], 1), dtype=int)
        self.cls_mask = np.ones((self.X.shape[0], 1), dtype=int)
        if continuous_mean_std is not None:
            self.mean, self.std = continuous_mean_std
        else:
            self.mean, self.std = np.array(self.Xcont,dtype=np.float32).mean(0), np.array(self.Xcont ,dtype=np.float32).std(0)
            self.std = np.where(self.std < 1e-6, 1e-6, self.std)
        
        if (imp_X is not None) and (t_mask is not None):
            self.imp_X = imp_X
            imp_X = np.array(self.imp_X.cpu())
            self.imp_Xcat = imp_X[:,self.cat_cols].copy().astype(np.int64) 
            self.imp_Xcon = imp_X[:,self.con_cols].copy().astype(np.float32) 
            if imp_mean_std is not None:
                self.imp_mean, self.imp_std = imp_mean_std
            else:
                self.imp_mean, self.imp_std = self.imp_Xcon.mean(0), self.imp_Xcon.std(0)
            self.imp_std = np.where(self.imp_std < 1e-6, 1e-6, self.imp_std)
            self.t_mask = t_mask
        else:
            self.imp_X = None
            self.imp_Xcat = None
            self.imp_Xcon = None
            self.imp_mean = None
            self.imp_std = None
            self.t_mask = None
    
       
    def generate_imputed(self, corruptor_settings, imp_mean_std=None):
        # mask generated from missing type imputation
        self.imp_X, self.t_mask  = imputed_data(self.X, corruptor_settings)
        imp_X = np.array(self.imp_X.cpu())
        self.imp_Xcat = imp_X[:,self.cat_cols].copy().astype(np.int64) #categorical columns
        self.imp_Xcon = imp_X[:,self.con_cols].copy().astype(np.float32) 
        if imp_mean_std is not None:
            self.imp_mean, self.imp_std = imp_mean_std
        else:
            self.imp_mean, self.imp_std = self.imp_Xcon.mean(0), self.imp_Xcon.std(0)
        self.imp_std = np.where(self.imp_std < 1e-6, 1e-6, self.imp_std)
        self.imp_Xcon = (self.imp_Xcon - self.imp_mean) / self.imp_std


    def __len__(self):
        # return len(self.y)
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        # return (
        #     np.concatenate((self.cls[idx], self.Xcat[idx])), self.Xcont[idx],
        #     np.concatenate((self.cls[idx], self.imp_Xcat[idx])), self.imp_Xcon[idx],
        #     self.y[idx], np.concatenate((self.cls_mask[idx], self.Xcat_mask[idx])), self.Xcont_mask[idx], self.t_mask[idx])   
        return (
            np.concatenate((self.cls[idx], self.Xcat[idx])), self.Xcont[idx],
            np.concatenate((self.cls[idx], self.imp_Xcat[idx])), self.imp_Xcon[idx],
            np.concatenate((self.cls_mask[idx], self.Xcat_mask[idx])), self.Xcont_mask[idx], self.t_mask[idx])   
    
class Dataset_test_imputed(Dataset):
    def __init__(self, imp_X, imp_mean_std=None, cat_cols=[]):
        self.imp_X = imp_X['data'].copy()

        self.cat_cols = list(cat_cols)
        self.con_cols = list(set(np.arange(self.imp_X.shape[1])) - set(cat_cols))

        self.imp_Xcat = imp_X[:,self.cat_cols].copy().astype(np.int64) 
        self.imp_Xcon = imp_X[:,self.con_cols].copy().astype(np.float32) 
        if imp_mean_std is not None:
            self.imp_mean, self.imp_std = imp_mean_std
        else:
            self.imp_mean, self.imp_std = self.imp_Xcon.mean(0), self.imp_Xcon.std(0)
        self.imp_std = np.where(self.imp_std < 1e-6, 1e-6, self.imp_std)
        self.imp_Xcon = (self.imp_Xcon - self.imp_mean) / self.imp_std

        self.cls = np.zeros((self.imp_X.shape[0], 1), dtype=int)
        self.cls_mask = np.ones((self.imp_X.shape[0], 1), dtype=int)

        self.X_mask = imp_X['mask'].copy()
        self.Xcat_mask = self.X_mask[:,self.cat_cols].copy().astype(np.int64)
        self.Xcont_mask = self.X_mask[:,self.con_cols].copy().astype(np.int64)


    def __len__(self):
        return self.imp_X.shape[0]
    
    def __getitem__(self, idx): 
        return (
            np.concatenate((self.cls[idx], self.imp_Xcat[idx])), self.imp_Xcon[idx],
            np.concatenate((self.cls_mask[idx], self.Xcat_mask[idx])), self.Xcont_mask[idx])   


def concat_data(X,y):
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

def X_split(X, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    return x_d


def data_prep_csv(filename, seed, categorical_indicator=None, y_column_name=None):
    print(f'dataseed = {seed}')
    
    np.random.seed(seed) 
    
    df = pd.read_csv(filename)
    all_columns = list(df.columns)
    if(y_column_name): 
        X = df[list(set(all_columns)-set([y_column_name]))]
        y = df[y_column_name]
    else:
        X = df[all_columns[:-1]]
        y = df[all_columns[-1]]
    
    if(not categorical_indicator):
        categorical_indicator = len(X.columns)*[False]

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    # print(cat_idxs, con_idxs, 'cat, con')
    for col in categorical_columns:
        X[col] = X[col].astype("object")

    y = y.values
    l_enc = LabelEncoder() 
    y = l_enc.fit_transform(y)
    row_idx = np.arange(X.shape[0])
    
    cv = StratifiedKFold(n_splits = 5, random_state=42, shuffle = True)
    k_folds_list = list(cv.split(row_idx, y))
    train_indices, test_indices = k_folds_list[seed]
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
        X = X.fillna(X.loc[train_indices, col].mean())
    
    X[cont_columns] = X[cont_columns].astype(np.uint8)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    train_mean_std = np.array([train_mean, train_std]).astype(np.float32)
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean_std


def data_prep_dataFrame(df, seed=0, con_idxs=None, cat_idxs=None, categorical_indicator=None):
    print(f'dataseed = {seed}')
    
    np.random.seed(seed) 
    X = df.copy()

    if(not cat_idxs):
        cat_idxs = []

    if(not con_idxs):
        con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    categorical_columns = np.array(X.columns)[cat_idxs].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
   
        
    # if(not categorical_indicator):
    #     categorical_indicator = len(X.columns)*[False]

    # categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    
    # cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    # cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    # con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    # print(cat_idxs, con_idxs, 'cat, con')
    for col in categorical_columns:
        X[col] = X[col].astype("object")

    row_idx = np.arange(X.shape[0])
    y = np.ones(len(row_idx))
    cv = StratifiedKFold(n_splits = 5, random_state=42, shuffle = True)
    k_folds_list = list(cv.split(row_idx, y))
    train_indices, test_indices = k_folds_list[seed]
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
        X = X.fillna(X.loc[train_indices, col].mean())
    
    X_train = X_split(X, nan_mask, train_indices)
    X_test = X_split(X, nan_mask, test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    return cat_dims, cat_idxs, con_idxs, X_train, X_test, train_mean, train_std



def generate_data_loader(X: np.array, imp_X: np.array, nan_mask: np.array, mean: np.array, std: np.array, cat_cols=[],  X_mask=None, create_ds=False, shuffle_data=True):
    ''''
    X: original data with mean on missing values
    imp_X: Imputed X (with NaNs or median values) 
    nan_mask: generated mask from masking method 
    mean: array with mean values of each column
    std: array with std values of each column
    cat_cols: indexes of categorical columns in X 
    create_ds: boolean to create or not the dataloader from a Dataset_imputed object (necessary in DeepIFSAC method)
    X_mask: mask of X, where 0 means the value was originally NaN and filled with the mean (necessary to create the Dataset_imputed object)
    shuffle_data: True if is X is training data and False if X is test data

    Output:
    data_loader: DataLoader object 
    '''
    torch.manual_seed(1)
    np.random.seed(1)

    con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
    X[:, con_cols] = standardize_data(X[:, con_cols], mean, std)
    imp_X[:,con_cols] = standardize_data(imp_X[:,con_cols], mean, std)

    if create_ds and (X_mask is not None):
        X_d = {
            'data': X,
            'mask': X_mask
        }
        
        imp_X =  torch.from_numpy(imp_X)
        t_mask =  torch.from_numpy(nan_mask)
        ds = Dataset_imputed(X_d, t_mask=t_mask, imp_X=imp_X, continuous_mean_std=(mean, std), imp_mean_std=(mean, std), cat_cols=cat_cols)
        data_loader = DataLoader(ds, batch_size=128, shuffle=shuffle_data, num_workers=0)
    else:
        data_loader = DataLoader(imp_X, batch_size=128, shuffle=shuffle_data, num_workers=0)
    return data_loader


def pre_process_deepifsac(X: pd.DataFrame, t_mask: np.array, median_features: np.array, con_idxs: list, cat_idxs=[]):
    ''''
    X: original data filled with mean on missing values
    t_mask: generated mask from masking method 
    median_features: array with median values of each column
    con_idxs: indexes of numerical columns in X
    cat_idxs: indexes of categorical columns in X 

    Output
    imp_X: X filled by median values according to t_mask
    '''
    imp_X = X.copy()

    for i, col in enumerate(imp_X.columns.values[con_idxs]):
        idxs = t_mask[:, i].nonzero()[0]
        imp_X.loc[idxs, col] = median_features[con_idxs[i]]
    
    cat_dims = np.append(np.array([1]), np.array(cat_idxs)).astype(int)
    X = X.values.astype(np.float32)
    imp_X = imp_X.values.astype(np.float32)
    return X, imp_X, cat_dims