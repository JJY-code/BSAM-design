# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


# set the random seed
def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    # if True,use same concolution algorithm every time
    torch.backends.cudnn.deterministic = True
    # for improve training speed Performance
    torch.backends.cudnn.benchmark = False
    # set numpy random seed
    np.random.seed(seed)
    # https://blog.csdn.net/qq_42951560/article/details/112174334
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Split
def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data_x into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    # the value of prediction
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        # https://zhuanlan.zhihu.com/p/509649964
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
            # Concatenate tensors https://pytorch.org/docs/stable/generated/torch.cat.html?highlight=torch%20cat#torch.cat
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim=1, layer_num=3, ):
        super(MyModel, self).__init__()
        # self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        # self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        # self.layer4 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
        hidden_layers = [128, 256, 128]
        layers = []
        in_features = int(in_dim)
        # 创建隐藏层
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            # layers.append(nn.BatchNorm1d(hidden_units))  # Batch Normalization
            layers.append(nn.Sigmoid())  # 可以根据需要更改激活函数
            in_features = hidden_units

        # # 创建隐藏层
        # for i, hidden_units in enumerate(hidden_layers):
        #     layers.append(nn.Linear(in_features, hidden_units))
        #     layers.append(nn.BatchNorm1d(hidden_units))  # Batch Normalization
        #     if i < len(hidden_layers) - 1:  # For all but the last hidden layer
        #         layers.append(nn.ReLU())  # 使用 ReLU 激活函数
        #     else:
        #         layers.append(nn.Sigmoid())  # 最后一层使用 Sigmoid 激活函数
        #     in_features = hidden_units

        # 创建输出层
        layers.append(nn.Linear(in_features, out_dim))
        self.model = nn.Sequential(*layers)


        # self.layer_num = layer_num
        # # , nn.BatchNorm1d(n_hidden_1)
        # # , nn.BatchNorm1d(n_hidden_1)
        # self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_neural_num),
        #                             nn.ReLU())  # , nn.BatchNorm1d(hidden_neural_num)
        # # self.layer2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
        # # self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
        # self.layer2 = nn.Sequential(nn.Linear(hidden_neural_num, hidden_neural_num), nn.ReLU())
        #
        # self.layer_last = nn.Sequential(nn.Linear(hidden_neural_num, out_dim))

    def forward(self, x):
        x = self.model(x)
        # x = self.layer1(x)
        # # x = self.layer2(x)
        # for i in range(self.layer_num):
        #     x = self.layer2(x)
        # x = self.layer_last(x)
        x = x.squeeze(1)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True, num_output_dims_fun=1):
    """Selects useful features to perform regression"""
    # https://blog.csdn.net/qq_40714949/article/details/127037956
    y_train, y_valid = train_data[:, -num_output_dims_fun:], valid_data[:,
                                                             -num_output_dims_fun:]  # the last Column is the label
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-num_output_dims_fun], valid_data[:,
                                                                                 :-num_output_dims_fun], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = list(range(35, raw_x_train.shape[1]))  # TODO: Select suitable feature_spiral columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


# # 学习率调度器
# def warmup_lr(step, model_size, warmup_steps):
#     return (model_size ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

# def my_cos_damping(step_lambda, warmup_steps, total_steps):
#     min(min((step_lambda+1)/warmup_steps, 1), math.cos((step_lambda+1-warmup_steps)/(total_steps-warmup_steps)))


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.

    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0)

    # Add warmup scheduler
    # 初始化warmup_steps和total_steps
    warmup_ratio = config['warmup_ratio']
    total_steps = len(train_loader) * config['n_epochs']
    warmup_steps = math.floor(total_steps * warmup_ratio)

    # power: # lambda step_lambda: min(min((step_lambda + 1) / warmup_steps, 1), ((step_lambda + 1) / (warmup_steps))**(-1))
    # liner: # lambda step_lambda: min(min((step_lambda+1)/warmup_steps, 1), (step_lambda+1-warmup_steps)/(total_steps-warmup_steps))
    # cos: # lambda step_lambda: min(min((step_lambda+1)/warmup_steps, 1), (math.cos((step_lambda+1-warmup_steps)/(total_steps-warmup_steps)*math.pi))/2+0.5)
    # exp: # lambda step_lambda: min(min((step_lambda + 1) / warmup_steps, 1), math.exp((1 - (step_lambda + 1) / warmup_steps) / 3))
    # const: # lambda step_lambda: 1.
    # warm_up: # lambda step_lambda: min((step_lambda + 1) / warmup_steps, 1)
    if config['warmup_flag'] == None:
        pass
    elif config['warmup_flag'] == 'torch_method':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],
                                                        total_steps=total_steps, pct_start=0.2)
    else:

        # my_liner_damping = lambda step_lambda: min(min((step_lambda+1)/warmup_steps, 1),
        #                                       math.cos((step_lambda+1-warmup_steps)/(total_steps-warmup_steps)))
        if config['warmup_flag'] == 'power':
            lam = lambda step_lambda: min(min((step_lambda + 1) / warmup_steps, 1),
                                          ((step_lambda + 1) / (warmup_steps)) ** (-1))
        elif config['warmup_flag'] == 'linear':
            lam = lambda step_lambda: min(min((step_lambda + 1) / warmup_steps, 1),
                                          (step_lambda + 1 - warmup_steps) / (total_steps - warmup_steps))
        elif config['warmup_flag'] == 'exp':
            lam = lambda step_lambda: min(min((step_lambda + 1) / warmup_steps, 1),
                                          math.exp((1 - (step_lambda + 1) / warmup_steps) / 3))
        elif config['warmup_flag'] == 'warm_up':
            lam = lambda step_lambda: min((step_lambda + 1) / warmup_steps, 1)
        elif config['warmup_flag'] == 'damping':
            lam = lambda step_lambda: min(1., math.exp((1 - (step_lambda + 1) / warmup_steps) / 3))

        else:
            raise Exception('warmup_flag error')

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam)

    writer = SummaryWriter()  # Writer of tensorboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data_x to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).

            # print
            # print(f'step:{step}\nlr:{optimizer.param_groups[0]['lr']}')

            # current_lr = optimizer.param_groups[0]['lr']
            optimizer.step()  # Update parameters.

            if config['warmup_flag'] is not None:
                # Update scheduler
                scheduler.step()

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, epoch + 1)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate/epoch', current_lr, epoch + 1)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        # print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, epoch + 1)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('\nSaving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


train_data_original, test_data = np.loadtxt('./dataset/train_data.txt'), np.loadtxt('./dataset/test_data.txt')
num_features_plus_num_predictions = train_data_original.shape[1]
num_features = test_data.shape[1]
num_output_dims = num_features_plus_num_predictions - num_features
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 520,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.1,  # validation_size = train_size * valid_ratio
    'n_epochs': 5000,  # Number of epochs.
    'batch_size': 128,
    'learning_rate': 5e-3,
    'early_stop': 500,  # If model has not improved for this many consecutive epochs, stop training.
    'warmup_flag': None,
    'warmup_ratio': 0.01,  # Add warmup steps in the configuration
    'num_features': num_features,
    'save_path': './models/model_rf_z.ckpt',  # Your model will be saved here.
    'load_model_path': None
}


def my_ann_main():
    # Set seed for reproducibility
    same_seed(config['seed'])
    train_data, valid_data = train_valid_split(train_data_original, config['valid_ratio'], config['seed'])
    # Print out the data_x size.
    # print(f"""train_data size: {train_data.shape}
    # valid_data size: {valid_data.shape}
    # test_data size: {test_data.shape}""")
    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'],
                                                             num_output_dims_fun=num_output_dims)

    train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
        COVID19Dataset(x_valid, y_valid), \
        COVID19Dataset(x_test)

    # Pytorch data_x loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    model = MyModel(in_dim=x_train.shape[1], out_dim=y_train.shape[1]).to(
        device)  # put your model and data_x on the same computation device.
    if config['load_model_path'] is not None:
        model.load_state_dict(torch.load(config['load_model_path'], weights_only=True))
    trainer(train_loader, valid_loader, model, config, device)
    model_pred = MyModel(in_dim=config['num_features'], out_dim=num_output_dims).to(device)
    model_pred.load_state_dict(torch.load(config['save_path'], weights_only=True))
    preds = predict(test_loader, model_pred, device)
    np.savetxt('./dataset/pred.txt', preds)


if __name__ == "__main__":
    my_ann_main()
