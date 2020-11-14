import torch


class Config:

    resume_training = False
    resume_from_fold = 1  # In case of k-fold training [1, k]

    loss_fn = ''
    metric = ''
    architecture = ''

    num_epochs = 200
    batch_size = 160
    learning_rate = 0.0003

    data_file = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
