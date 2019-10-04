import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cols = ['Date', 'Close', 'Open', 'Volume', 'High', 'Low']
cols = ['Date', 'Close', 'Volume']
short_time = 12
long_time = 100
# time_prev = 100
time_prev = 64

num_future_days = 5
# i: i + num_future_days 828

lstm_params = {
    "input_dim": None,
    "hidden_dim": 128,
    "num_layer": 1,
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "output_dim": num_future_days,
    "seq_len": time_prev,
    "bidirectional": False
}
