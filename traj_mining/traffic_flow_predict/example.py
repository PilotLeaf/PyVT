import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
from matplotlib.ticker import MaxNLocator
import generate_dataset
import lstm_encoder_decoder
import plotting
import torch

matplotlib.rcParams.update({'font.size': 14})
print(torch.cuda.is_available())

# generate dataset for LSTM
t, y, xorigin, yorigin = generate_dataset.read_data('./data/result.csv', 'total')
t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, y, split=0.80)

fig1, ax1 = plt.subplots(1, figsize=(12, 6))
ax1.plot(t, y, linewidth=1, marker='o', ls='-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Hourly traffic volume')
ax1.yaxis.set_major_locator(MaxNLocator(10))
ax1.xaxis.set_major_locator(MaxNLocator(7))

fig1.autofmt_xdate()
ax1.grid(True)
plt.savefig('plots/time_series.png')

# plot time series with train/test split
fig2, ax2 = plt.subplots(1, figsize=(12, 6))
ax2.plot(t_train, y_train, linewidth=1, ls='-', marker='o', label='Train')
ax2.plot(np.concatenate([[t_train[-1]], t_test]), np.concatenate([[y_train[-1]], y_test]), linewidth=1, ls='--',
         marker='*', label='Test')
ax2.set_xlabel('Time')
ax2.set_ylabel('Hourly traffic volume')
plt.title('Time Series Split into Train and Test Sets')
plt.legend()
plt.tight_layout
plt.savefig('plots/train_test_split.png')

# set size of input/output windows 
iw = 24
ow = 6
s = 3

# generate windowed training/test datasets
Xtrain, Ytrain = generate_dataset.windowed_dataset(y_train, input_window=iw, output_window=ow, stride=s)
Xtest, Ytest = generate_dataset.windowed_dataset(y_test, input_window=iw, output_window=ow, stride=s)
# Xtimetrain, Ytimetrain = generate_dataset.windowed_dataset(t_train, input_window=iw, output_window=ow, stride=s)
# Xtimetest, Ytimetest = generate_dataset.windowed_dataset(t_test, input_window=iw, output_window=ow, stride=s)

# plot example of windowed data
fig3, ax3 = plt.subplots(1, figsize=(12, 6))
ax3.plot(np.arange(0, iw), Xtrain[:, 0, 0], ls='-', linewidth=1.0, marker='o', label='Input')
ax3.plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, 0, 0]], Ytrain[:, 0, 0]]), linewidth=1.0,
         ls='--', marker='o', label='Target')
plt.xlim([0, iw + ow - 1])
# plt.yticks(np.arange(0, 24, 4))
ax2.set_xlabel('Time')
ax2.set_ylabel('Hourly traffic volume')
plt.title('Example of Windowed Training Data')
plt.legend()
plt.tight_layout()
plt.savefig('plots/windowed_data.png')

# LSTM encoder-decoder
# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# specify model parameters and train
import torch

ispre = True  # 是否预测
if ispre:
    # 加载模型
    m_state_dict = torch.load('./model/lstm-ed.pt')
    new_m = lstm_encoder_decoder.lstm_seq2seq(input_size=X_train.shape[2], hidden_size=128)
    new_m.load_state_dict(m_state_dict)
    plotting.plot_train_test_results(new_m, Xtrain, Ytrain, Xtest, Ytest)
    plt.close('all')
else:
    # specify model parameters and train
    # 保存模型
    model = lstm_encoder_decoder.lstm_seq2seq(input_size=X_train.shape[2], hidden_size=128)
    loss = model.train_model(input_tensor=X_train, target_tensor=Y_train, n_epochs=300, target_len=ow, batch_size=64,
                             training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.5,
                             learning_rate=0.001,
                             dynamic_tf=False)
    torch.save(model.state_dict(), './model/lstm-ed.pt')
