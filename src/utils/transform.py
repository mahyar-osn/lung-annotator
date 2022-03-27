import torch


def feature_transform_regularizer(trans, device):
    d = trans.size()[1]
    batch_size = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        # I = I.cuda()
        I = I.to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
