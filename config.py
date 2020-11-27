import torch


class Config:
    data_folder_path = './IAM/'  # relative to ./data/
    img_h = 32
    char_w = 16
    partition = 'tr'

    batch_size = 16
    num_epochs = 200
    epochs_lr_decay = 100  # learning rate decay will be applied for last these many steps (should be <= num_epochs)
    resume_training = False

    train_gen_steps = 4  # generator weights to be updated after every specified number of steps
    grad_alpha = 1
    grad_balance = True

    data_file = './data/tr_data.pkl'
    lexicon_file = './data/words.txt'

    architecture = 'ScrabbleGAN'
    # Recognizer network
    r_ks = [3, 3, 3, 3, 3, 3, 2]
    r_pads = [1, 1, 1, 1, 1, 1, 0]
    r_fs = [64, 128, 256, 256, 512, 512, 512]

    # Generator and Discriminator networks
    # arch[g_resolution] defines the architecure to be selected
    # arch[16] has been added in BigGAN.py with parameters as specified in the paper
    resolution = 16
    bn_linear = 'SN'
    g_shared = False

    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'

    # Noise vector
    z_dim = 128
    num_chars = 74