import torch
from torch import nn
from preprocessing import train_loader, test_loader, scaler, scaler_labels
from configs import *
# from preprocessing import *
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


class LSTMNet(nn.Module):

    def __init__(self,
                 _input_dim,
                 _hidden_dim,
                 _output_dim,
                 _batch_size,
                 _device,
                 _num_layer=1,
                 _bidirectional=False):
        super(LSTMNet, self).__init__()
        self.input_dim = _input_dim
        self.hidden_dim = _hidden_dim
        self.output_dim = _output_dim
        self.batch_size = _batch_size
        self.device = _device
        self.num_layer = _num_layer
        self.bidirectional = _bidirectional

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layer,
                            bidirectional=bidirectional)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_dim),
            nn.LeakyReLU(0.01)
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
model = LSTMNet(_input_dim=input_dim,
                _hidden_dim=hidden_dim,
                _output_dim=output_dim,
                _batch_size=batch_size,
                _device=device).to(device)

# define loss function
criterion = nn.MSELoss()

# define opimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train():
    for epoch in range(num_epochs):

        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            features = features.view(seq_len, -1, input_dim)

            labels = labels.to(device).float()
            labels = labels.view(1, -1, output_dim)
            # compute forward
            outs = model(features)
            # print("outs: ", outs.size())

            # compute loss
            loss = criterion(outs, labels)

            # compute gradients
            optimizer.zero_grad()
            loss.backward()

            # update parameters
            optimizer.step()

        if epoch > 10:
            if epoch % 10 == 0:
                print("Epoch {}/{} -- Loss: {:.5}"
                      .format(epoch, num_epochs, loss.item()))
        else:
            print("Epoch {}/{} -- Loss: {:.5}"
                  .format(epoch + 1, num_epochs, loss.item()))
    torch.save(model.state_dict(), './model.ckpt')


def load_model():
    _model = LSTMNet(_input_dim=input_dim,
                    _hidden_dim=hidden_dim,
                    _output_dim=output_dim,
                    _batch_size=batch_size,
                    _device=device).to(device)
    _model.load_state_dict(torch.load('./model.ckpt'))
    print("Eval: ", model.eval())
    return _model


# """
def test(model):
    with torch.no_grad():
        l = 0
        total = 0
        for features, true_labels in test_loader:
            total += len(true_labels) * 5
            true_labels = true_labels.to(device).float()
            true_labels = true_labels.view(1, -1, output_dim)
            features = features.view(seq_len, -1, input_dim)
            preds = model(features)
            preds = preds.view(preds.size(1) * preds.size(2))
            true_labels = true_labels.view(true_labels.size(1) * true_labels.size(2))

            converted = scaler_labels.inverse_transform(preds.reshape(-1, 1))
            true = scaler_labels.inverse_transform(true_labels.reshape(-1, 1))
            print("invert: \n", converted.reshape(-1, output_dim))
            print("true label: \n", true.reshape(-1, output_dim))

            # print("labels: \n", labels)
            # print("out pred: \n", preds)

            ab = np.abs(preds - true_labels) / true_labels
            # print("ab: ", ab)
            ab = ab.numpy()
            # print(type(ab))
            l += np.sum(ab, axis=0)
            print("l: ", l)

            # print("----: \n", np.sum(np.abs(preds - labels)/labels))
            # loss = criterion(preds, labels)
            # total += loss
        print("total: ", total)
        mape = l / total

        print('Error with test: {}'.format(mape))
# """

# train()
# #
# load_model()

test(load_model())
