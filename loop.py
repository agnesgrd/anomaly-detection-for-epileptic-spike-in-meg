import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch.nn as nn

import numpy as np

# from torch.utils.data import Dataset, DataLoader
from statistics import mean
from utils import initialize_train_metrics, initialize_val_metrics, set_train_metrics, set_val_metrics, get_mean_metrics
import params
from thop import profile
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    loss_train = []

    for data in dataloader:
        inputs, targets = data

        inputs = inputs.to(device)
        inputs = inputs.float()

        outputs = model(inputs)

        targets = targets.to(device)
        targets = targets.float()

        targets = targets.squeeze(dim=1)
        outputs = outputs.squeeze(dim=1)

        try:
            loss = criterion(outputs, targets)
            loss_train.append(float(loss))

        except RuntimeError as e:
            print(f"Skipping batch due to error: {e}")
            continue
        except IndexError as i:
            print(f"Skipping batch due to error: {i}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Delete unnecessary variables to free memory
        del inputs, targets, outputs, loss
    return mean(loss_train)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    loss_valid = []

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data

            inputs = inputs.to(device)
            inputs = inputs.float()

            outputs = model(inputs)

            targets = targets.to(device)
            targets = targets.float()

            targets = targets.squeeze()

            inputs = inputs.squeeze()
            outputs = outputs.squeeze()

            loss = criterion(outputs, targets)
            loss_valid.append(float(loss))

            #loss_valid.append(validation_loss.item())
        # Delete unnecessary variables to free memory
        del inputs, targets, outputs, loss
    return mean(loss_valid)

def test_mega_epoch(model, dataloader, criterion, device, test = False):
        model.eval()
        target_all = []
        output_all = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                input, target = data

                input = input.to(device)

                output = model(input)
                if not test:
                    output = (output>0.5).float()
                elif test:
                    output = (output>0.15).float()

                target = target.squeeze(dim=1).cpu().numpy()
                output = output.squeeze(dim=1).cpu().numpy()

                # target = np.argmax(target, axis=1)
                # output = np.argmax(output, axis=1)

                target_all.append(target)
                output_all.append(output)

            target = np.hstack(target_all)
            output = np.hstack(output_all)

            return target, output

def train_mega_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    train_metrics = initialize_train_metrics()
    data_rec_losses = []
    flag = 1
    for data in tqdm(dataloader):
        input, target = data

        input = input.to(device)
        idx = torch.arange(274).to(device)

        # profile
        # if flag == 1:
        #     hereflops, parameters = profile(model, (input, idx, device))
        #     print("flops and params", hereflops, parameters)
        #     flag = 0

        output = model(input, idx, device)
        
        rec_loss = params.a*criterion(output[0][0], output[0][1])
        +params.b*criterion(output[1][0], output[1][1])
        +params.c*criterion(output[2][0], output[2][1])
        +params.d*criterion(output[3][0], output[3][1])
        +params.e*criterion(output[4][0], output[4][1])
        
        data_rec_losses.append(rec_loss.item())
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()
        train_metrics = set_train_metrics(train_metrics, rec_loss)
    train_metrics = get_mean_metrics(train_metrics)
    scheduler.step()
    # utils.log_train_metrics(train_metrics, epoch)
    return train_metrics['rec_loss']

def validate_mega_epoch(model, dataloader, criterion, device):
    model.eval()
    data_rec_losses = []
    val_metrics = initialize_val_metrics()
    with torch.no_grad():
        for data in tqdm(dataloader):
            input, target = data

            input = input.to(device)
            idx = torch.arange(274).to(device)

            output = model(input, idx, device)

            target = target.to(device)
            target = target.float()

            rec_loss = params.a*criterion(output[0][0], output[0][1])
            +params.b*criterion(output[1][0], output[1][1])
            +params.c*criterion(output[2][0], output[2][1])
            +params.d*criterion(output[3][0], output[3][1])
            +params.e*criterion(output[4][0], output[4][1])

            data_rec_losses.append(rec_loss.item())
            val_metrics = set_val_metrics(val_metrics, rec_loss)

        val_metrics = get_mean_metrics(val_metrics)

        return val_metrics['rec_loss']

    





