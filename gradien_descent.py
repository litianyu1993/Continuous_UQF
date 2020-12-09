import torch
from torch import nn
import torch.nn.functional as F

def train(model, device, train_loader, optimizer):
    error = []
    for batch_idx, (x, target) in enumerate(train_loader):
        action, obs, target = x[0].to(device), x[1].to(device), target.to(device)
        x = [action, obs]
        optimizer.zero_grad()
        output = model(x).to(device)
        #print(output.shape, target.shape)
        loss = F.mse_loss(output, target)
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
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

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
            print('Epoch: '+str(epoch)+' Train Error: {:.4f} Validate Error: {:.4f}'.format(train_loss, validate_loss))
        scheduler.step()
        train_loss_vec.append(train_loss)
        validate_loss_vec.append(validate_loss)
        # print(model.action_encoder.encoder[0].weight)
        # print(model.obs_encoder.encoder[0].weight)
        # print(model.mps[0])

    return train_loss_vec, validate_loss_vec