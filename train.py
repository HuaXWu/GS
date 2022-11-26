import os

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import loadData
import loadDataOrigin
import net_model
import model
from torch.utils.tensorboard import SummaryWriter
import utils
import CNN_model
# from utils import create_lr_scheduler, get_params_groups

writer = SummaryWriter(log_dir="./runs/CNN/")


"""
 train function.
 choice the dataloader and model by yourself.
"""
def train_func(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:8833/')
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    # load data
    x_train, x_test, x_valid, y_train, y_test, y_valid = loadDataOrigin.load_data(random_state=args.rds,
                                                                            label_col_name=args.label)
    train_loader, test_loader = loadDataOrigin.to_tensor_loader(x_train, x_test, y_train, y_test)
    valid_loader = loadDataOrigin.load_valid_data(x_valid, y_valid)
    net = model.resnet10(num_classes=args.num_classes)

    # model_weight_path = args.weights
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # dummy_input = torch.rand(1, 3, 653, 424)
    # with SummaryWriter(comment="CNN-4") as w:
    #     w.add_graph(net, dummy_input)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if net.state_dict()[k].numel() == v.numel()}
            print(net.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # freeze  weight.
    if args.freeze_layers:
        for name, para in net.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("train {}".format(name))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, args.num_classes)
    net.to(device)
    print(net)
    params = [p for p in net.parameters() if p.requires_grad]
    # params = get_params_groups(net)
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
    #                                    warmup=True, warmup_epochs=1)

    # method one : decay by steps
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    # method two : decay by cosine curve
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    lr_scheduler = utils.create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)
    all_scheduler = []
    # method three decay by loss
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_acc = 0.0
    save_path = args.save_patch
    all_train_loss = []
    all_train_corr = []
    all_test_loss = []
    all_test_corr = []
    all_valid_loss = []
    all_valid_corr = []
    for epoch in range(args.epochs):
        # scheduler.step()
        all_scheduler.append(optimizer.param_groups[0]["lr"])
        train_loss, train_acc, train_val = utils.train_one_epoch(model=net,
                                                                 optimizer=optimizer,
                                                                 data_loader=train_loader,
                                                                 device=device,
                                                                 epoch=epoch, lr_scheduler=lr_scheduler)
        print("=====train_avg_loss is {}, train_avg_corr is {} train_val is {}=====".
              format(train_loss, train_acc, train_val))

        # test
        test_loss, test_acc, test_val = utils.train_one_epoch(model=net,
                                                              optimizer=optimizer,
                                                       data_loader=test_loader,
                                                       device=device,
                                                       epoch=epoch, lr_scheduler=lr_scheduler
                                                       )
        print("=====test_avg_loss is {}, test_avg_corr is {} test_val is{}=====".
              format(test_loss, test_acc, test_val))

        # valid

        val_loss, val_acc, valid_variance = utils.evaluate(model=net,
                                                           data_loader=valid_loader,
                                                           device=device,
                                                           epoch=epoch,
                                                           all_epoch=args.epochs
                                                           )

        lr_scheduler.step()
        all_train_loss.append(train_loss.item())
        all_train_corr.append(train_acc.item())
        all_test_loss.append(test_loss.item())
        all_test_corr.append(test_acc.item())
        all_valid_loss.append(val_loss.item())
        all_valid_corr.append(val_acc.item())
        print("=====valid_avg_loss is {}, valid_avg_corr is {}, valid_variance is {}=====".
              format(val_loss, val_acc, valid_variance))

        tags = ["train_loss", "train_acc", "test_loss", "test_acc", "train_var", "test_var", "valid_loss", "valid_acc",
                "valid_var", "learning_rate"]
        writer.add_scalar(tags[0], train_loss, epoch)
        writer.add_scalar(tags[1], train_acc, epoch)
        writer.add_scalar(tags[2], test_loss, epoch)
        writer.add_scalar(tags[3], test_acc, epoch)
        # writer.add_scalar(tags[5], test_var, epoch)
        # writer.add_scalar(tags[6], valid_loss, epoch)
        # writer.add_scalar(tags[7], valid_acc, epoch)
        # writer.add_scalar(tags[8], valid_var, epoch)
        # writer.add_scalar(tags[4], train_var, epoch)
        writer.add_scalar(tags[9], optimizer.param_groups[0]["lr"], epoch)

        # save the pth
        # best_corr = val_acc
        # if best_corr > best_acc:
        #     best_acc = best_corr
        #     torch.save(net.state_dict(), save_path)

    print("train{}\n{}\ntest{}\n{}\nvalid{}\n{}\n".format(
        all_train_loss, all_train_corr, all_test_loss, all_test_corr, all_valid_loss, all_valid_corr))
    print(all_scheduler)
    print("Finish train")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.001)

    # pre-train pth
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--wd', type=float, default=5e-3)
    parser.add_argument('--rds', type=int, default=12)
    parser.add_argument('--save_patch', type=str, default="./weights/resnext50_32x4d-pig_bf_01_tt.pth")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--label', type=str, default="label")
    opt = parser.parse_args()

    train_func(opt)
    # valid_func(opt)
