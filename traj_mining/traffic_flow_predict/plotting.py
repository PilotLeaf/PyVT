import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import datetime
import matplotlib.ticker as ticker
import sklearn.metrics as metrics


def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, num_rows=3):
    '''
    plot examples of the lstm encoder-decoder evaluated on the training/test data

    : param lstm_model:     trained lstm encoder-decoder
    : param Xtrain:         np.array of windowed training input data
    : param Ytrain:         np.array of windowed training target data
    : param Xtest:          np.array of windowed test input data
    : param Ytest:          np.array of windowed test target data
    : param num_rows:       number of training/test examples to plot
    : return:               num_rows x 2 plots; first column is training data predictions,
    :                       second column is test data predictions
    '''

    # input window size
    iw = Xtrain.shape[0]
    ow = Ytest.shape[0]

    # figure setup
    num_cols = 2
    num_plots = num_rows * num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(13, 15))

    # plot training/test predictions
    for ii in range(num_rows):
        # train set
        X_train_plt = Xtrain[:, ii, :]
        Y_train_pred = lstm_model.predict(torch.from_numpy(X_train_plt).type(torch.Tensor), target_len=ow)

        ax[ii, 0].plot(np.arange(0, iw), Xtrain[:, ii, 0], 'k', linewidth=2, label='Input')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Ytrain[:, ii, 0]]),
                       color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Y_train_pred[:, 0]]),
                       color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        ax[ii, 0].set_xlim([0, iw + ow - 1])
        ax[ii, 0].set_xlabel('$t$')
        ax[ii, 0].set_ylabel('$y$')

        # test set
        X_test_plt = Xtest[:, ii, :]
        Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len=ow)
        ax[ii, 1].plot(np.arange(0, iw), Xtest[:, ii, 0], 'k', linewidth=2, label='Input')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Ytest[:, ii, 0]]),
                       color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Y_test_pred[:, 0]]),
                       color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        ax[ii, 1].set_xlim([0, iw + ow - 1])
        ax[ii, 1].set_xlabel('$t$')
        ax[ii, 1].set_ylabel('$y$')

        # plot_results(np.arange(0, iw), Xtrain[:, ii, 0], np.arange(iw - 1, iw + ow),
        #              np.concatenate([[Xtrain[-1, ii, 0]], Ytrain[:, ii, 0]]), np.arange(iw - 1, iw + ow),
        #              np.concatenate([[Xtrain[-1, ii, 0]], Y_train_pred[:, 0]]), 'train' + str(ii))
        plot_results(np.arange(0, iw), Xtest[:, ii, 0],
                     np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Ytest[:, ii, 0]]),
                     np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Y_test_pred[:, 0]]),
                     'test' + str(ii))

        if ii == 0:
            ax[ii, 0].set_title('Train')
            ax[ii, 1].legend(bbox_to_anchor=(1, 1))
            ax[ii, 1].set_title('Test')

    plt.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('plots/predictions.png')
    # plt.show()
    plt.close()

    return


def plot_results(xi, yi, xt, yt, xp, yp, tpe):
    fig1, ax1 = plt.subplots(1, figsize=(12, 6))
    yi = yi * 32
    yt = yt * 32
    yp = yp * 32
    ax1.plot(['2019-4-1 0:00',
              '2019-4-1 1:00',
              '2019-4-1 2:00',
              '2019-4-1 3:00',
              '2019-4-1 4:00',
              '2019-4-1 5:00',
              '2019-4-1 6:00',
              '2019-4-1 7:00',
              '2019-4-1 8:00',
              '2019-4-1 9:00',
              '2019-4-1 10:00',
              '2019-4-1 11:00',
              '2019-4-1 12:00',
              '2019-4-1 13:00',
              '2019-4-1 14:00',
              '2019-4-1 15:00',
              '2019-4-1 16:00',
              '2019-4-1 17:00',
              '2019-4-1 18:00',
              '2019-4-1 19:00',
              '2019-4-1 20:00',
              '2019-4-1 21:00',
              '2019-4-1 22:00',
              '2019-4-1 23:00'], yi, linewidth=1, marker='o', ls='-', label='Input')
    ax1.plot(['2019-4-1 23:00',
              '2019-4-2 0:00',
              '2019-4-2 1:00',
              '2019-4-2 2:00',
              '2019-4-2 3:00',
              '2019-4-2 4:00',
              '2019-4-2 5:00'], yt, linewidth=1, marker='o', ls='-', label='Target')
    ax1.plot(['2019-4-1 23:00',
              '2019-4-2 0:00',
              '2019-4-2 1:00',
              '2019-4-2 2:00',
              '2019-4-2 3:00',
              '2019-4-2 4:00',
              '2019-4-2 5:00'], yp, linewidth=1, marker='*', ls='--', label='Prediction')

    mape = metrics.mean_absolute_percentage_error(yt, yp)
    vs = metrics.explained_variance_score(yt, yp)
    mae = metrics.mean_absolute_error(yt, yp)
    mse = metrics.mean_squared_error(yt, yp)
    r2 = metrics.r2_score(yt, yp)
    rmse = np.sqrt(mse)
    print('mape:' + str(mape))
    print('mae:' + str(mae))
    print('mse:' + str(mse))
    print('r2:' + str(r2))
    print('rmse:' + str(rmse))

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Hourly traffic volume')
    ax1.yaxis.set_major_locator(MaxNLocator(10))
    ax1.xaxis.set_major_locator(MaxNLocator(12))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylim(0, 35)
    fig1.autofmt_xdate(rotation=30)
    ax1.grid(True)
    plt.legend()
    plt.title('Vessel traffic flow prediction')
    plt.tight_layout()
    plt.savefig('plots/' + str(tpe) + '.png')
    return
