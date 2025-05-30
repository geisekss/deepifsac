from models.model import *
from metrics.loss import calculate_constrative_loss, calculate_denoising_loss
import os, pickle
import torch.optim as optim
from corruptor import Corruptor
from utils import embed_data_mask

class DeepIFSAC(nn.Module):

    """
    DeepIFSAC Model
    ------------------

    This module implements the DeepIFSAC model, an adaptation of the SAINT model for tabular data.

    Original SAINT Reference:
        "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training"

    Original implementation available at:
        https://github.com/somepago/saint
    """
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        corruptor_settings,
        dim = 32,
        depth = 6,
        heads = 8,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 0,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'colrow',
        final_mlp_style = 'sep'
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        # categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        # categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        # self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            #col : Self-attention only
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow', 'rowcol','parallel', 'colrowatt'] :
            # row: between-attention, colrow: self-between-withContrastive-attention, rowcol: between-self-withContrastive-attention, colrowatt: self-between-attention
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        # Q: What are these masks for?
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        
        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            # map each feature embeddings to either categorical or numerical
            # i.e. reconstruction from feature embeddings
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))

        # self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.corruptor_settings = corruptor_settings


    def forward(self, x_categ, x_cont):  
        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:,:self.num_categories,:])
        con_outs = self.mlp2(x[:,self.num_categories:,:])
        return cat_outs, con_outs 
    

    def fit(self, trainloader, filename_metrics, epochs, fold_key, device=0):
         if os.path.exists(filename_metrics):
            with open(filename_metrics, 'rb') as f:
                try:
                    metrics_dict = pickle.load(f)
                except EOFError:
                    metrics_dict = {}
         else:
            metrics_dict = {}
         missing_rate_key = f'missing_{self.corruptor_settings['missing']}'
         metrics_dict.setdefault(missing_rate_key, {})
         metrics_dict[missing_rate_key].setdefault(fold_key, {'epochs': {}})
         optimizer = optim.AdamW(self.parameters(), lr=0.0001)
         criterion1 = nn.CrossEntropyLoss()
         corruptor = Corruptor(trainloader.dataset.imp_X, self.corruptor_settings)
         print("DeepIFSAC training begins!")

         for epoch in range(epochs):
            running_loss, num_batches = 0.0, 0
            self.train()
            for batch in trainloader:
                optimizer.zero_grad()
                # Unpack and move batch to device
                # x_categ, x_cont, x_categ_imp, x_cont_imp, _, cat_mask, con_mask, train_mask_batch = [d.to(device) for d in batch]
                x_categ, x_cont, x_categ_imp, x_cont_imp, cat_mask, con_mask, train_mask_batch = [d.to(device) for d in batch]
            
                x_cont_corr = corruptor(x_cont_imp)
                x_categ_corr = x_categ_imp
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(self, x_categ_corr, x_cont_corr, cat_mask, con_mask)
                _, x_categ_enc, x_cont_enc = embed_data_mask(self, x_categ, x_cont, cat_mask, con_mask)

                # Contrastive loss (only one block is retained)
                loss = calculate_constrative_loss(self, x_categ_enc, x_cont_enc, x_categ_enc_2, x_cont_enc_2, criterion1)
        
                # Denoising loss
                loss += calculate_denoising_loss(self, x_categ, x_cont, x_categ_enc_2, x_cont_enc_2, train_mask_batch, criterion1)
            
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_batches += 1

            # Record metrics for the epoch
            metrics_dict[missing_rate_key][fold_key]['epochs'][f'epoch_{epoch}'] = {'running_loss': running_loss}
            print(f'Epoch {epoch + 1}, Loss: {running_loss / num_batches}')

         with open(filename_metrics, 'wb') as f:
            pickle.dump(metrics_dict, f)


    def transform(self, data_loader, device='cpu', mean_std=None, imp_mean_std=None):
        all_predictions = []
        all_original_data = []
        all_predictions_cat = []
        all_original_cat = []
        y_list = []
        mask_t = []
        self.eval()
        with torch.no_grad():
            for data in data_loader:
                # x_categ, x_cont, x_categ_imp, x_cont_imp, y_t, cat_mask, con_mask, t_mask = [d.to(device) for d in data]
                x_categ, x_cont, x_categ_imp, x_cont_imp, cat_mask, con_mask, t_mask = [d.to(device) for d in data]
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(self, x_categ_imp, x_cont_imp, cat_mask, con_mask)
                cat_outs, con_outs = self(x_categ_enc_2, x_cont_enc_2)
                con_outs = [x.cpu().numpy() for x in con_outs]
                cat_outs_device = [x.cpu().numpy() for x in cat_outs]
                mask_t.append(t_mask.cpu().numpy())
                # y_list.append(y_t.cpu().numpy())
                all_predictions.append(np.concatenate(con_outs, axis=1))
                all_original_data.append(x_cont.cpu().numpy())
                all_predictions_cat.append(np.concatenate(cat_outs_device, axis=1))
                all_original_cat.append(x_categ.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_original_data = np.concatenate(all_original_data, axis=0)
        all_predictions_cat = np.concatenate(all_predictions_cat, axis=0)
        all_original_cat = np.concatenate(all_original_cat, axis=0)
        mask_t = np.concatenate(mask_t, axis=0)
        observed_entries = 1 - mask_t
        
        mean, std = mean_std
        mean_imp, std_imp = imp_mean_std
        all_predictions = torch.tensor((all_predictions * std_imp) + mean_imp)
        all_original_data = torch.tensor((all_original_data * std) + mean)
        all_predictions = (all_original_data * observed_entries) + (all_predictions * mask_t)

        mse_con = torch.mean((all_original_data - all_predictions) ** 2, dim=0)
        feature_variances = torch.where(torch.var(all_original_data, dim=0) == 0,
                                        torch.ones_like(torch.var(all_original_data, dim=0)),
                                        torch.var(all_original_data, dim=0))
        nrmse_con = torch.mean(torch.sqrt(mse_con) / feature_variances)
        return(all_predictions, nrmse_con)