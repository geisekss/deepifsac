import torch

import numpy as np
import pandas as pd
from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling

# need to remove unwanted imports

default_settings = {
    'method': 'pass',          # 'pass', 'noise' | 'draw' | 'sample' | 'knn' | 'mice'
    'corruption_rate': .6,      # 0.6 or between 0-1; fraction of features to corrupt (not used for mice/knn)
    'missing': .2,              # 0.2 between 0-1 float;  % of missingness
    'missing_type': 'mcar',     # 'mcar' | 'mnar' | 'mar'
    'mice': 'LinearRegression', # 'LinearRegression' | 'DecisionTree' | others...
}
# mask_arr = None

class Corruptor:

    def __init__(self, X_original, settings, mask=None):
        '''
        X_orginal = Full (train/valid) features (needed for sampling/drawing)
        settings = dictionary of settings (see default settings)

        '''
        # overwrite keys provided on default settings
        settings = {**default_settings, **settings}
        # print(settings)
        self.method = settings['method']
        self.corruption_rate = settings['corruption_rate']
        self.X_original = X_original
        self.missing = settings['missing']
        self.mask = mask
        sampler_map = {
            'mnar': mnar_sampling,
            'mcar': mcar_sampling,
            'mar': mar_sampling,
        }
        self.missing_type = settings['missing_type']
        self.missing_sampler = sampler_map[self.missing_type]
        self.mice = settings['mice']
        
        
    def _get_mask(self, X):
        '''
        TODO: implement without for-loop
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X=X.to(device)
        n,d = X.shape
        # debug_mode and print(X.shape)
        d_corrupt = int(self.corruption_rate * d)
        x = np.zeros((n,d))

        for i in range(n):
            a = np.arange(1,d+1)
            a1 = np.random.permutation(a)
            x[i,:] = a1

        mask = np.where(x<=d_corrupt, 1, 0)

        device = X.device
        mask = torch.from_numpy(mask)
        mask = mask.to(device)
        
        return mask
    
    def _get_nan_mask(self, X):
        '''
        TODO: implement without for-loop
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        n,d = X.shape
        # debug_mode and print(X.shape)
        d_corrupt = int(self.corruption_rate * d)
        x = np.zeros((n,d))

        for i in range(n):
            a = np.arange(1,d+1)
            a1 = np.random.permutation(a)
            x[i,:] = a1

        mask = np.where(x, 1, 0)

        # to( = X.to(device)
        mask = torch.from_numpy(mask)
        
        # mask = mask if to(device)<0 else mask.to(to(device))
        mask = mask.to(device)
        # debug_mode and print('mask shape', mask.shape)
        
        return mask
    
    def _get_c_mask(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = X.clone().to(device)
        nan_mask = torch.where(torch.isnan(X), torch.tensor(1).to(device), torch.tensor(0).to(device))
        return nan_mask
    
    
    def _mcar_missing(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        X1 = torch.from_numpy(X_missing.to_numpy())
        self.mask = self._get_c_mask(X1)
        
        return X1, self.mask

    def _cutmix(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.clone(X0).to(device)
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        
        X1 = torch.from_numpy(X_missing.to_numpy())
        mask = self._get_c_mask(X1)
        mask = mask.to(device)
        
        # select random rows for each row (can have same row idx)
        r = torch.randint(self.X_original.shape[0],(X.shape[0],))
        noise_values = self.X_original[r,:].to(device)
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)
        x_cont_missing = (real + draws).detach().clone()
        
        return x_cont_missing  
    
    def _draw(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.clone(X0).to(device)
        _, mask = self._mcar_missing(X)
        mask = mask.to(device)

        # select random rows for each row (can have same row idx)
        r = torch.randint(self.X_original.shape[0],(X.shape[0],))
        noise_values = self.X_original[r,:].to(device)
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return torch.tensor(real + draws), mask

    def _draw_error(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.clone(X0).to(device)
        # to(device) = X.to(device)
        mask = self._get_c_mask(X)
        
        # select random rows for each row (can have same row idx)
        r = torch.randint(self.X_original.shape[0],(X.shape[0],))
        noise_values = self.X_original[r,:]

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        # print('DRAW:   ',real+draws)

        return real + draws
    
    def _draw_ichi(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        imputed_tensor = X.clone().to(device)
        nan_indices = torch.isnan(imputed_tensor)
        random_values = torch.randn_like(imputed_tensor)
        
        # Replace NaN values with random values
        imputed_tensor[nan_indices] = random_values[nan_indices]
        
        # print('Draw:', imputed_tensor)
        # return imputed_tensor.cpu().numpy()
        return imputed_tensor

    def _drawX(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        imputed_tensor = X.clone().to(device)

        # Get the indices of nan values using the mask
        nan_indices = torch.where(torch.isnan(X), torch.tensor(1), torch.tensor(0)).bool()

        # Perform random draw imputation
        for indices in zip(*nan_indices):
            # Get random indices from the original data shape
            random_indices = tuple(torch.randint(imputed_tensor.size(dim), (1,)) for dim in imputed_tensor.shape)

            # Replace the nan values with random values from the original data
            imputed_tensor[indices] = imputed_tensor[random_indices]

        print("Draw: ", imputed_tensor)

        return imputed_tensor
    

    def _nanstd(self, x, mean):     
        epsilon = 1e-8
        return torch.sqrt(torch.nanmean(torch.pow(x - mean, 2) + epsilon, dim=-1))
    

    def _sample(self, X0):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Get the shape of the input tensor
        X = torch.clone(X0).to(device)
        shape = X.shape
        mask = self.mask.bool()
        noise_values = torch.empty_like(X).normal_()

        # Apply the nan_mask to select the noise values where NaN values are present
        imputed_values = torch.where(mask.to(device), noise_values.to(device), self.X_original.to(device))
        # print("sample:   ",imputed_values)
        return imputed_values
    
    def __call__(self, X):
        
        method_map = {
            'pass': lambda x: x,
            'cutmix': self._cutmix,
            'sample': self._sample,
            'draw': self._draw,
            'mcar_missing' : self._mcar_missing,
        }
        
        return method_map[self.method](X)