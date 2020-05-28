from dataset import ModisDataset, Sentinel5Dataset
import torch
import numpy as np
from tqdm.autonotebook import tqdm
import os
import matplotlib.pyplot as plt
from model import Model, snapshot, restore
import ignite.metrics
import pandas as pd
from dataset import transform_data

N_SEEN_POINTS = 227
N_PREDICTIONS = 50

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_layers = 3
    hidden_size = 32
    region = "canada"#"volcanopuyehue" #"germany"
    epochs = 100
    include_time = True
    smooth = None
    use_attention = True

    model_dir="/data2/igarss2020/models/"
    log_dir = "/data2/igarss2020/models/"
    name_pattern = "LSTM_{region}_l={num_layers}_h={hidden_size}_e={epoch}"
    log_pattern = "LSTM_{region}_l={num_layers}_h={hidden_size}"

    model = Model(input_size=1 if not include_time else 2,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  output_size=1,
                  dropout=0.5,
                  device=device)

    #restore("/data2/igarss2020/models/LSTM_germany_l=1_h=128_e=2.pth", model)

    #model.load_state_dict(torch.load("/tmp/model_epoch_0.pth")["model"])
    model.train()

    remove_seasonality = False

    if True:
        dataset = ModisDataset(region=region,
                               fold="train",
                               znormalize=True,
                               augment=True,
                               overwrite=False,
                               include_time=include_time,
                               filter_date=(None,"2010-01-01"),
                               remove_seasonality=remove_seasonality,
                               smooth=smooth)

        validdataset = ModisDataset(region=region,
                                    fold="validate",
                                    znormalize=True,
                                    znormalize_with_meanstd=(dataset.mean, dataset.std),
                                    augment=False,
                                    filter_date=(None, None),
                                    include_time=include_time,
                                    remove_seasonality=remove_seasonality,
                                    smooth=smooth)

    else:
        dataset = Sentinel5Dataset(fold="train", seq_length=300, include_time=include_time)
        validdataset = Sentinel5Dataset(fold="validate", seq_length=300, include_time=include_time)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=512,
                                             shuffle=True,
                                             #sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(10000))
                                             )
    validdataloader = torch.utils.data.DataLoader(validdataset,
                                             batch_size=512,
                                             shuffle=False,
                                             #sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(10000))
                                             )

    #criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    L2Norm = torch.nn.MSELoss(reduction='none')
    def criterion(y_pred, y_data, log_variances):
        norm = (y_pred-y_data)**2
        loss = (torch.exp(-log_variances) * norm).mean()
        regularization = log_variances.mean()
        return 0.5 * (loss + regularization)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    cutoff_after_n_epochs=2
    loss_thresshold = -1

    stats=list()
    for epoch in range(epochs):

        trainloss = train_epoch(model,dataloader,optimizer, criterion, device, loss_thresshold)
        testmetrics, testloss = test_epoch(model,validdataloader,device, criterion, n_predictions=1)
        metric_msg = ", ".join([f"{name}={metric.compute():.2f}" for name, metric in testmetrics.items()])
        msg = f"epoch {epoch}: train loss {trainloss.mean():.2f}, test loss {testloss:.2f}, {metric_msg}"
        print(msg)

        idx = trainloss.sort()[1][int(len(trainloss) * 0.6)]
        loss_thresshold = trainloss[idx]

        test_model(model, validdataset, device, title=f"                 epoch {epoch}")

        model_name = name_pattern.format(region=region, num_layers=num_layers, hidden_size=hidden_size, epoch=epoch)
        pth = os.path.join(model_dir, model_name+".pth")
        print(f"saving model snapshot to {pth}")
        snapshot(model, optimizer, pth)

        stat = dict()
        stat["epoch"] = epoch
        for name, metric in testmetrics.items():
            stat[name]=metric.compute()

        stat["trainloss"] = trainloss.cpu().detach().numpy()
        stat["testloss"] = testloss.cpu().detach().numpy()
        stats.append(stat)

    df = pd.DataFrame(stats)

def train_epoch(model, dataloader, optimizer, criterion, device, loss_threshold=None):
    model.dropout.eval()
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    losses = list()
    for idx, batch in iterator:
        x_data, y_true = batch

        x_data = x_data.to(device)
        y_true = y_true.to(device)

        if y_true.shape[2] == 2:
            doy = x_data[:,:,1]
            y_true = y_true[:, :, 0].unsqueeze(2)
        else:
            doy = None

        optimizer.zero_grad()
        # y is used for teacher forcing during training
        y_pred, log_variances = model(x_data, y=y_true, date=doy)
        loss = criterion(y_pred, y_true, log_variances)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach())
        iterator.set_description(f"loss {loss:.4f}, mean log_variances {log_variances.mean():.6f}")

    return torch.stack(losses)

def test_epoch(model,dataloader, device, criterion, n_predictions):
    # switch on dropout

    metrics = dict(
        mae=ignite.metrics.MeanAbsoluteError(),
        mse=ignite.metrics.MeanSquaredError(),
        rmse=ignite.metrics.RootMeanSquaredError()
    )

    losses = list()

    with torch.no_grad():
        iterator = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in iterator:
            x_data, y_true = batch

            x_data = x_data.to(device)
            y_true = y_true.to(device)

            if y_true.shape[2] == 2:
                doy = x_data[:, :, 1]
                y_true = y_true[:, :, 0].unsqueeze(2)
            else:
                doy = None

            # make single forward pass to get test loss
            model.dropout.eval()
            y_pred,log_variances = model(x_data, date=doy)
            loss = criterion(y_pred, y_true, log_variances)
            losses.append(loss.cpu())

            model.dropout.train()
            # make multiple MC dropout inferences for further metrics
            y_pred, epi_var, ale_var = model.predict(x_data, n_predictions, date=doy)
            for name, metric in metrics.items():
                metric.update((y_pred.view(-1), y_true.view(-1)))

    return metrics, torch.stack(losses).mean()

def test_model(model, dataset, device, title=""):
    from visualizations import make_and_plot_predictions

    if isinstance(dataset,ModisDataset):

        idx = 1

        x = dataset.data[idx].astype(float)
        date = dataset.date[idx].astype(np.datetime64)
        N_seen_points = N_SEEN_POINTS
        N_predictions = N_PREDICTIONS

        make_and_plot_predictions(model, x, date, N_seen_points=N_seen_points, N_predictions=N_predictions,
                                  device=device,meanstd=(dataset.mean,dataset.std), title=f"Example {idx}" + title)
        plt.show()

        idx = 17

        x = dataset.data[idx].astype(float)
        date = dataset.date[idx].astype(np.datetime64)
        make_and_plot_predictions(model, x, date, N_seen_points=N_seen_points, N_predictions=N_predictions,
                                  device=device,meanstd=(dataset.mean,dataset.std), title=f"Example {idx}")
        plt.show()

    elif isinstance(dataset,Sentinel5Dataset):
        #d, date = dataset.data["Napoli"]
        d, date = dataset.get_data("Napoli")
        d = d.swapaxes(0,1)
        #x = d[:,:,0]
        #d[:, :, 0] = dataset.znormalize(d[:, :, 0])
        #date = d[:, :, 0].astype(np.datetime64)

        d[:,:,0] = d[:,:,0] - dataset.mean
        d[:, :, 0] = d[:,:,0] / dataset.std


        make_and_plot_predictions(model, d[0], date, N_seen_points=N_SEEN_POINTS, N_predictions=N_PREDICTIONS,
                                  device=device)
        plt.show()

    else:
        return


    """
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    
    future = x.shape[0] - N_seen_points


    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(date[:N_seen_points], x[:N_seen_points,0], c="#000000", alpha=1, label="seen input sequence")
    ax.plot(date[N_seen_points:], x[N_seen_points:,0], c="#000000", alpha=.1, label="unseen future")

    for idx in range(N_predictions):
        model.train()
        x_data = torch.Tensor(x)[None, :N_seen_points].to(device)
        y_pred, _ = model(x_data, future=future)
        y_pred = np.append([np.nan], y_pred[0, :-1, 0].cpu().detach().numpy())
        label = "prediction" if idx == 0 else None
        ax.plot(date[:N_seen_points + future], y_pred, c="#0065bd", label=label, alpha=(1 / N_predictions) ** 0.7)

    ax.axvline(x=date[N_seen_points], ymin=0, ymax=1)
    ax.legend()
    plt.show()
    """


def fine_tune(x, model, criterion, optimizer, inner_steps=1, device=torch.device('cpu')):
    model.lstm.flatten_parameters()

    for i in range(inner_steps):
        model.zero_grad()
        x_data, y_true = transform_data(x[:, None, :], seq_len=100)
        x_data = x_data.to(device)
        y_true = y_true.to(device)

        y_pred, log_variances = model(x_data)
        loss = criterion(y_pred, y_true, log_variances)
        loss.backward()
        optimizer.step()
    return model

if __name__ == '__main__':
    main()
