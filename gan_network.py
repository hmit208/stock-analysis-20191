import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from pytorch.advance.gan_network.gan_stock.preprocessing import train_loader, test_loader, scaler, scaler_labels
from pytorch.advance.gan_network.gan_stock.configs import *
# from pytorch.advance.gan_network.gan_stock.preprocessing import *
import numpy as np
import random

device = device
input_dim = lstm_params['input_dim']
hidden_dim = lstm_params['hidden_dim']
num_layer = lstm_params['num_layer']
num_epochs = lstm_params['num_epochs']
batch_size = lstm_params['batch_size']
learning_rate = lstm_params['learning_rate']
output_dim = lstm_params['output_dim']
seq_len = lstm_params['seq_len']
bidirectional = lstm_params['bidirectional']
torch.manual_seed(9999)
np.random.seed(9999)
random.seed(9999)


class DiscriminatorNet(nn.Module):

    def __init__(self, n_features, n_out):
        super(DiscriminatorNet, self).__init__()
        self.n_features = n_features
        self.n_out = n_out

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 72),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(72, 100),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(100, 10),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(10, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        out = self.out(x)

        return out


class GeneratorNet(nn.Module):

    def __init__(self,
                 _input_dim,
                 _hidden_dim,
                 _output_dim,
                 _batch_size,
                 _device,
                 _num_layer=1,
                 _bidirectional=False):
        super(GeneratorNet, self).__init__()
        self.input_dim = _input_dim
        self.hidden_dim = _hidden_dim
        self.output_dim = _output_dim
        self.batch_size = _batch_size
        self.device = _device
        self.num_layer = _num_layer
        self.bidirectional = _bidirectional

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layer,
                            bidirectional=self.bidirectional)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_dim),
            # nn.LeakyReLU(0.01),
            nn.Sigmoid()
        )

    def init_hidden(self):
        # 1*1 = num_layers*directions

        h0 = torch.zeros(self.num_layers * 1, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers * 1, self.batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_dim).to(self.device)
        c0 = torch.zeros(1, x.size(1), self.hidden_dim).to(self.device)

        on, (hn, cn) = self.lstm(x.float(), (h0, c0))
        out = self.out(hn)
        return out


# define model
generator = GeneratorNet(_input_dim=input_dim,
                         _hidden_dim=hidden_dim,
                         _output_dim=output_dim,
                         _batch_size=batch_size,
                         _device=device).to(device)

discriminator = DiscriminatorNet(n_features=output_dim, n_out=1)

loss = nn.BCELoss()

# lựa chọn optimizer
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data.to(device)


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data.to(device)


def train_discriminator(optimizer, real_data, fake_data):
    # reset gradient
    optimizer.zero_grad()

    # training on real data
    prediction_real = discriminator(real_data)

    # calculate error and backward
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # training on fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # update params
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    prediction = prediction.view(prediction.size(1), -1)
    # print("predictionasdasdasd :", prediction)
    # print("sdasdasd: ", real_data_target(prediction.size(0)))
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    optimizer.step()
    return error


for epoch in range(num_epochs):
    for n_batch, (features, real_data) in enumerate(train_loader):
        # 1. training discriminator
        # generate fake data
        features = features.view(seq_len, -1, input_dim)

        fake_data = generator(features.float()).detach()
        # print("noise(real_batch.size(0)): ", noise(real_batch.size(0)))
        # Train D
        fake_data = fake_data.view(len(real_data), num_future_days)
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data.float(),
                                                                fake_data.float())

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(features)
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log error

        # Display Progress
    if epoch % 10 == 0:
        print("Epoch {} - Gloss: {} - Dloss: {}".format(epoch, g_error, d_error))
            # print("fake_data: \n", fake_data)
            # print("real_data: \n", real_data)
            # print("d_pred_fake: \n", d_pred_fake)
            # print("d_pred_real: \n", d_pred_real)
        # Model Checkpoints
        # logger.save_models(generator, discriminator, epoch)
