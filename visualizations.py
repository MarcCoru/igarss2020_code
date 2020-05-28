import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import torch

tumblack = "#000000"
tumblue = "#0065bd"
tumorange = "#e37222"
tumbluelight = "#64a0c8"
tumgray = "#999999"
tumlightgray = "#dad7cb"

def make_and_plot_predictions(model, x, date, N_seen_points=250, N_predictions=50, ylim=None, device=torch.device('cpu'), store=None, meanstd=None, title=""):

    future = x.shape[0] - N_seen_points

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(title)
    axs = np.array(axs).reshape(-1)

    axs[0].set_title("epistemic uncertainty")
    axs[1].set_title("aleatoric uncertainty")
    axs[2].set_title("combined uncertainty")

    x_ = torch.Tensor(x)[None, :].to(device)
    if x_.shape[2] == 2:
        doy_seen = x_[:, :N_seen_points, 1]
        doy_future = x_[:, N_seen_points:, 1]
    else:
        doy_seen = None
        doy_future=None
    x_data = x_[:, :N_seen_points, 0].unsqueeze(2)

    mean, epi_var, ale_var,y_hat = model.predict(x_data, N_predictions, future, date=doy_seen, date_future=doy_future, return_yhat=True)
    var = epi_var + ale_var

    mean = mean.cpu().squeeze()
    var = var.cpu().squeeze()
    epi_var = epi_var.cpu().squeeze()
    ale_var = ale_var.cpu().squeeze()

    epi_std = torch.sqrt(epi_var[1:])
    ale_std = torch.sqrt(ale_var[1:])
    data_std = epi_std + ale_std

    if meanstd is not None:
        dmean, dstd = meanstd
        x = (x * dstd) + dmean
        mean = (mean * dstd) + dmean
        y_hat = (y_hat * dstd) + dmean
        ale_std = ale_std * dstd
        epi_std = epi_std * dstd
        data_std = data_std * dstd

    n_sigma = 1
    axs[0].fill_between(date[1:], mean[1:] + epi_std, mean[1:] - epi_std,
                        alpha=.5, label=f"epistemic {n_sigma}" + r"$\sigma$")
    axs[1].fill_between(date[1:], mean[1:] + ale_std, mean[1:] - ale_std,
                        alpha=.5, label=f"aleatoric {n_sigma}" + r"$\sigma$")
    axs[2].fill_between(date[1:], mean[1:] + data_std, mean[1:] - data_std, alpha=.5,
                        label=r"combined $\sigma$")

    for ax in axs:
        ax.plot(date[:N_seen_points], x[:N_seen_points, 0], c="#000000", alpha=1, label="seen input sequence")
        ax.plot(date[N_seen_points:], x[N_seen_points:, 0], c="#000000", alpha=.1, label="unseen future")
        if N_seen_points < len(date):
            ax.axvline(x=date[N_seen_points], ymin=0, ymax=1)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(date[1:], mean[1:],c=tumorange)

    #for y_pred in y_hat:
    #    label = "prediction" if idx == 0 else None
    #    ax.plot(date[:N_seen_points + future], y_pred, c=tumorange, label=label, alpha=(1 / N_predictions) ** 0.7)

    [ax.legend(ncol=3) for ax in axs]

    if store is not None:
        import pandas as pd
        df = pd.DataFrame([date, mean.numpy(), epi_std.numpy(), ale_std.numpy(), data_std.numpy(), x[:, 0]],
                     index=["date", "mean", "epi_std", "ale_std", "std", "x"]).T
        df["mean-epistd"] = df["mean"] - df["epi_std"]
        df["mean+epistd"] = df["mean"] + df["epi_std"]
        df["mean-alestd"] = df["mean"] - df["ale_std"]
        df["mean+alestd"] = df["mean"] + df["ale_std"]
        df["mean-std"] = df["mean"] - df["std"]
        df["mean+std"] = df["mean"] + df["std"]
        df.iloc[:N_seen_points].to_csv(f"{store}_seen.csv")
        df.iloc[N_seen_points:].to_csv(f"{store}_predicted.csv")
        df.to_csv(f"{store}.csv")
        print(f"saving to {store}")
        
        preds = pd.DataFrame(y_hat.squeeze().cpu().numpy(), index=[f"pred{run}" for run in range(N_predictions)]).T
        preds["date"] = date
        preds.to_csv(f"{store}_predictions.csv")

    return fig, axs

def make_and_plot_combined_predictions(model, x, date, N_seen_points=250, N_predictions=50, ylim=None,
                              device=torch.device('cpu'), store=None, meanstd=None, ax=None):
    future = x.shape[0] - N_seen_points

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 3))

    x_ = torch.Tensor(x)[None, :].to(device)
    if x_.shape[2] == 2:
        doy_seen = x_[:, :N_seen_points, 1]
        doy_future = x_[:, N_seen_points:, 1]
    else:
        doy_seen = None
        doy_future = None
    x_data = x_[:, :N_seen_points, 0].unsqueeze(2)

    mean, epi_var, ale_var, y_hat = model.predict(x_data, N_predictions, future, date=doy_seen, date_future=doy_future,
                                                  return_yhat=True)
    var = epi_var + ale_var

    mean = mean.cpu().squeeze()
    var = var.cpu().squeeze()
    epi_var = epi_var.cpu().squeeze()
    ale_var = ale_var.cpu().squeeze()

    epi_std = torch.sqrt(epi_var[1:])
    ale_std = torch.sqrt(ale_var[1:])
    data_std = epi_std + ale_std

    if meanstd is not None:
        dmean, dstd = meanstd
        x = (x * dstd) + dmean
        mean = (mean * dstd) + dmean
        y_hat = (y_hat * dstd) + dmean
        ale_std = ale_std * dstd
        epi_std = epi_std * dstd
        data_std = data_std * dstd

    ax.fill_between(date[1:], mean[1:] + data_std, mean[1:] - data_std, alpha=.5,
                        label=r"combined $\sigma$")

    ax.plot(date[:N_seen_points], x[:N_seen_points, 0], c="#000000", alpha=1, label="seen input sequence")
    ax.plot(date[N_seen_points:], x[N_seen_points:, 0], c="#000000", alpha=.1, label="unseen future")
    ax.axvline(x=date[N_seen_points], ymin=0, ymax=1)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.plot(date[1:], mean[1:], c=tumorange)

    if store is not None:
        import pandas as pd
        df = pd.DataFrame([date, mean.numpy(), epi_std.numpy(), ale_std.numpy(), data_std.numpy(), x[:, 0]],
                          index=["date", "mean", "epi_std", "ale_std", "std", "x"]).T
        df["mean-epistd"] = df["mean"] - df["epi_std"]
        df["mean+epistd"] = df["mean"] + df["epi_std"]
        df["mean-alestd"] = df["mean"] - df["ale_std"]
        df["mean+alestd"] = df["mean"] + df["ale_std"]
        df["mean-std"] = df["mean"] - df["std"]
        df["mean+std"] = df["mean"] + df["std"]
        df.iloc[:N_seen_points].to_csv(f"{store}_seen.csv")
        df.iloc[N_seen_points:].to_csv(f"{store}_predicted.csv")
        df.to_csv(f"{store}.csv")
        print(f"saving to {store}")

        preds = pd.DataFrame(y_hat.squeeze().cpu().numpy(), index=[f"pred{run}" for run in range(N_predictions)]).T
        preds["date"] = date
        preds.to_csv(f"{store}_predictions.csv")

    return ax

def predict_future(model, x, future, date, N_predictions=50, ax=None, device=torch.device('cpu'), meanstd=None):

    if ax is None:
        fig,ax = plt.subplots(figsize=(12,4))

    x_ = torch.Tensor(x)[None, :].to(device)
    x_data = x_[:, :, 0].unsqueeze(2)

    dt = np.median(np.diff(date))
    future_dates = date[-1] + np.arange(1,future+1)*dt
    all_dates = np.hstack([date, future_dates])

    mean, epi_var, ale_var,y_hat = model.predict(x_data, N_predictions, future, return_yhat=True)

    var = epi_var + ale_var

    mean = mean.cpu().squeeze()
    var = var.cpu().squeeze()
    epi_var = epi_var.cpu().squeeze()
    ale_var = ale_var.cpu().squeeze()

    epi_std = torch.sqrt(epi_var[1:])
    ale_std = torch.sqrt(ale_var[1:])
    data_std = epi_std + ale_std

    if meanstd is not None:
        dmean, dstd = meanstd
        x = (x * dstd) + dmean
        mean = (mean * dstd) + dmean
        y_hat = (y_hat * dstd) + dmean
        ale_std = ale_std * dstd
        epi_std = epi_std * dstd
        data_std = data_std * dstd

    ax.plot(all_dates[1:],mean[1:].cpu().numpy())
    ax.plot(date,x[:,0])
    ax.axvline(x=date[-1])
    ax.fill_between(all_dates[1:], mean[1:] + data_std, mean[1:] - data_std, alpha=.5)