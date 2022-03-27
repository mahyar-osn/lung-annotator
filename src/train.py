import os
import json
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optimiser
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from core.dataset import LungDataset
from core.model import PointNetDenseCls
from utils.transform import feature_transform_regularizer


__DEVICE__ = "cuda" if torch.cuda.is_available() else "cpu"
__PIN_MEMORY__ = True if __DEVICE__ == "cuda" else False


class ProgramArguments(object):
    def __init__(self):
        self.input_config = None


def _prepare_data(config: dict):
    root = config["root"]
    path = os.path.join(root, config["lung"])

    points_path = list()
    for file_name in os.listdir(os.path.join(path, "points")):
        points_path.append(os.path.join(path, "points", file_name))

    labels_path = list()
    for file_name in os.listdir(os.path.join(path, "points_label")):
        labels_path.append(os.path.join(path, "points_label", file_name))

    split = train_test_split(points_path,
                             labels_path,
                             test_size=config["split"],
                             random_state=42)

    train_points, test_points = split[:2]
    train_labels, test_labels = split[2:]

    train_dataset = LungDataset(train_points, train_labels)
    test_dataset = LungDataset(test_points, test_labels)

    # TODO: Suspected PyTorch bug when try to pin the memory:
    # A leaking CAFFE2 thread-pool after fork. Possibly doesn't really matter but let's ignore the ping_memory for now.

    # create the training data loaders
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=config["batch_size"],
                              # pin_memory=__PIN_MEMORY__,
                              num_workers=config["workers"])
    # create the test data loaders
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=config["batch_size"],
                             # pin_memory=__PIN_MEMORY__,
                             num_workers=config["workers"])

    return train_dataset, test_dataset, train_loader, test_loader


def train(config: dict, train_loader: DataLoader, test_loader: DataLoader, train_dataset: LungDataset):
    blue_log = lambda x: '\033[94m' + x + '\033[0m'

    num_classes = 3
    model = PointNetDenseCls(k=num_classes, feature_transform=config["feature_transform"])
    optimizer = optimiser.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optimiser.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # model.cuda()
    model.to(__DEVICE__)
    num_batch = len(train_dataset) // config["batch_size"]

    for epoch in range(config["epoch"]):
        scheduler.step()
        for i, data in enumerate(train_loader, 0):
            points, target = data
            points = points.transpose(2, 1)
            # points, target = points.cuda(), target.cuda()
            points, target = points.to(__DEVICE__), target.to(__DEVICE__)
            optimizer.zero_grad()
            classifier = model.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            if config["feature_transform"] == "True":
                loss += feature_transform_regularizer(trans_feat, __DEVICE__) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print(f"[{epoch}: {i} / {num_batch}] train loss: {loss.item()} "
                  f"accuracy: {correct.item() / float(config['batch_size'] * 2500)}")

            if i % 10 == 0:
                j, data = next(enumerate(test_loader, 0))
                points, target = data
                points = points.transpose(2, 1)
                # points, target = points.cuda(), target.cuda()
                points, target = points.to(__DEVICE__), target.to(__DEVICE__)
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                b = blue_log('test')
                print(f"[{epoch}: {i} / {num_batch}] {b} loss: {loss.item()} "
                      f"accuracy: {correct.item() / float(config['batch_size'] * 2500)}")

        torch.save(model.state_dict(), f"{config['output_model']}/seg_model_{epoch}.pth")

    shape_ious = []
    for i, data in tqdm(enumerate(test_loader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        # points, target = points.cuda(), target.cuda()
        points, target = points.to(__DEVICE__), target.to(__DEVICE__)
        classifier = model.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)  # np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print(f"mIOU: {np.mean(shape_ious)}")


def main():
    args = parse_args()
    if os.path.exists(args.input_config):
        with open(args.input_config, "r") as input_config:
            config = json.load(input_config)

        train_data, test_data, train_loader, test_loader = _prepare_data(config)

        print(f"[INFO] found {len(train_data)} samples in the training set...")
        print(f"[INFO] found {len(test_data)} samples in the test set...")

        train(config, train_loader, test_loader, train_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Training a PointNet architecture on 3D lung data cloud.")
    parser.add_argument("input_config", help="Location of the configuration file.")

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


if __name__ == '__main__':
    main()
