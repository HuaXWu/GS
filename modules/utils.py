import torch
from tqdm import tqdm
import sys
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import math
import json


# MSE


# 交叉熵


def train_one_epoch_classify(model, optimizer, data_loader, device, epoch, lr_scheduler):
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        optimizer.zero_grad()
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        # update lr
        lr_scheduler.step(loss)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,0


@torch.no_grad()
def evaluate_classify(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,0


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    loss_function = torch.nn.MSELoss()
    model.train()
    step = 0
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_corr = torch.zeros(1).to(device)  # 累计准确率
    var_info = 0  # 差的方差
    train_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        x_train, y_train = data
        legit = model(x_train.to(device))
        y_train = y_train.to(device)
        loss = loss_function(legit, y_train)
        loss.backward()
        running_loss = loss.item()
        accu_loss += running_loss
        y_train = y_train.data.cpu().numpy().flatten()
        y_predict = legit.data.cpu().numpy().flatten()
        var = __variance__(y_train, y_predict)
        var_info += var
        corr = np.corrcoef(y_train, y_predict)[0, 1]
        accu_corr += corr.item()
        train_bar.desc = "train epoch[{} loss:{:.3f} corr is {:.3f} var_info is {:.3f},lr: {:.5f}".format(epoch + 1,
                                                                                                          running_loss,
                                                                                                          corr, var,
                                                                                                          optimizer.param_groups[
                                                                                                              0][
                                                                                                              "lr"])

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
    return accu_loss / (step + 1), accu_corr / (step + 1), var_info/ (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, all_epoch):
    loss_function = torch.nn.MSELoss()
    model.eval()
    step = 0
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_corr = torch.zeros(1).to(device)  # 累计准确率
    var_info = 0  # 差的方差
    valid_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(valid_bar):
        x_test, y_test = data
        y_predict = model(x_test.to(device))
        loss = loss_function(y_predict, y_test.to(device))
        accu_loss += loss.item()
        y_test = y_test.data.cpu().numpy().flatten()
        y_predict = y_predict.cpu().data.numpy().flatten()
        var = __variance__(y_test, y_predict)
        var_info += var
        corr = np.corrcoef(y_test, y_predict)[0, 1]
        accu_corr += corr.item()
        valid_bar.desc = "valid epoch[{}/{}] loss:{:.3f} corr is {:.3f} var_info is {:.3f}".format(epoch + 1,
                                                                                                   all_epoch, loss,
                                                                                                   corr, var)

        # early_stopping(loss, model)
        # # 若满足 early stopping 要求
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        # # 获得 会自动加载结果最好时候的模型参数。
        # model.load_state_dict(torch.load('checkpoint.pt'))
    return accu_loss / (step + 1), accu_corr / (step + 1), var_info/ (step + 1),


def __variance__(y_label, y_predict):
    info = y_label.reshape((-1, 1)) - y_predict.reshape((-1, 1))
    return np.var(info)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


