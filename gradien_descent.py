import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch import optim

def log_mse(output, target):
    loss = torch.log(torch.mean((output - target) ** 2))
    #loss = torch.log(torch.mean((torch.log(output) - torch.log(target)) ** 2))
    return loss

def MAPE(output, target):
    loss = torch.mean(torch.abs(target - output)/target)
    # print(torch.abs(target - output)[:5], target.shape)
    #print(torch.mean(torch.abs(target - output)/target))
    #print(torch.abs(output - target)[0], torch.abs(target)[0], output[0], target[0])
    return loss


def train(model, device, train_loader, optimizer):
    error = []
    for batch_idx, (x, target) in enumerate(train_loader):
        action, obs, target = x[0].to(device), x[1].to(device), target.to(device)
        x = [action, obs]
        optimizer.zero_grad()
        output = model(x).to(device)
        #print(output.shape, target.shape)
        #loss = F.mse_loss(output, target)
        loss = log_mse(output, target)
        loss.backward()
        optimizer.step()
        error.append(loss.item())
        #print(model.action_encoder.encoder[0].weight)
    print(output[0], target[0])

    return sum(error) / len(error)


def validate(model, device, test_loader):
    #hankel.eval()
    test_loss = 0
    with torch.no_grad():
        for x, target in test_loader:
            action, obs, target = x[0].to(device), x[1].to(device), target.to(device)
            x = [action, obs]
            output = model(x).to(device)
            #test_loss += F.mse_loss(output, target).item()  # sum up batch loss
            #test_loss += log_mse(output, target).item()
            test_loss += MAPE(output, target).item()
            #print(MAPE(output, target).item())
    # print(output[:5])
    # print(target[:5])
    test_loss /= len(test_loader)

    return test_loss

def train_validate (model, train_lambda, validate_lambda, scheduler, option):
    option_default = {
        'verbose': True,
        'epochs': 10000
    }
    option = {**option_default, **option}
    train_loss_vec = []
    validate_loss_vec = []
    for epoch in range(1, option['epochs'] + 1):
        train_loss = train_lambda(model)
        validate_loss = validate_lambda(model)
        if option['verbose']:
            print('Epoch: '+str(epoch)+'Train Error: {:.4f} Validate Error: {:.4f}'.format(train_loss, validate_loss))
        scheduler.step()
        train_loss_vec.append(train_loss)
        validate_loss_vec.append(validate_loss)

    return train_loss_vec, validate_loss_vec

def fit(model, train_lambda, validate_lambda, **option):
    option_default = {
        'step_size': 500,
        'gamma': 0.1,
        'epochs': 1000,
        'verbose': True,
        'optimizer': optim.Adam(model.parameters(), lr=0.001, amsgrad=True),
    }
    option = {**option_default, **option}

    scheduler_params = {
        'step_size': option['step_size'],
        'gamma': option['gamma']
    }
    train_option = {
        'epochs': option['epochs'],
        'verbose': option['verbose']
    }
    scheduler = StepLR(option['optimizer'], **scheduler_params)
    model.fit(train_lambda, validate_lambda, scheduler, **train_option)
    return model