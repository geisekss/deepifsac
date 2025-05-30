import torch
import torch.nn.functional as F

NCE_TEMP = 0.7
LAM0 = 0.5
LAM1 = 10
LAM2 = 1
LAM3 = 10

def calculate_constrative_loss(model, x_categ_enc, x_cont_enc, x_categ_enc_2, x_cont_enc_2, criterion1):
    aug1 = model.transformer(x_categ_enc, x_cont_enc)
    aug2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
    aug1 = (aug1 / aug1.norm(dim=-1, keepdim=True)).flatten(1, 2)
    aug2 = (aug2 / aug2.norm(dim=-1, keepdim=True)).flatten(1, 2)

    aug1 = model.pt_mlp(aug1)
    aug2 = model.pt_mlp2(aug2)

    logits1 = aug1 @ aug2.t() / NCE_TEMP
    logits2 = aug2 @ aug1.t() / NCE_TEMP
    targets = torch.arange(logits1.size(0)).to(logits1.device)
    loss = LAM0 * (criterion1(logits1, targets) + criterion1(logits2, targets)) / 2
    return loss

def calculate_denoising_loss(model, x_categ, x_cont, x_categ_enc_2, x_cont_enc_2, train_mask_batch, criterion1):
    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
    if con_outs:
        con_outs = torch.cat(con_outs, dim=1)
        # if opt.have_xOrg:
        #     l2 = criterion2(con_outs, x_cont)
        # else:
        #     l2 = F.mse_loss(con_outs * (1 - train_mask_batch), x_cont * (1 - train_mask_batch), reduction='none')
        #     N = (1 - train_mask_batch).sum()
        #     l2 = l2.sum() / N
        #not opt.have_xOrg
        l2 = F.mse_loss(con_outs * (1 - train_mask_batch), x_cont * (1 - train_mask_batch), reduction='none')
        N = (1 - train_mask_batch).sum()
        l2 = l2.sum() / N
    else:
        l2 = 0
    l1 = sum(criterion1(cat_outs[j], x_categ[:, j]) for j in range(1, x_categ.shape[-1]))
    loss = LAM2 * l1 + LAM3 * l2
    return loss

 