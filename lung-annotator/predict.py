import os
import argparse
import warnings
from sys import platform

import torch
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from core.model import PointNetDenseCls


__DEVICE__ = "cpu"


class ProgramArguments(object):
    def __init__(self):
        self.input_file = None
        self.model = None
        self.feature = None


def show_prediction(points, prediction):
    from utils.show3d import show_points
    cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    pred_color = cmap[prediction.numpy()[0], :]
    show_points(points, c_pred=pred_color)


def save_prediction(points: np.array, prediction: np.array, output_path: str) -> None:
    output = np.hstack((points, prediction.T))
    np.save(output_path.replace(".pts", "_pred.npy"), output.astype(np.float16))
    return None


def main():
    args = parse_args()
    if os.path.exists(args.input_file):
        point_set = np.loadtxt(args.input_file).astype(np.float32)
        denorm_point_set = point_set.copy()
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        points = torch.from_numpy(point_set)
        state_dict = torch.load(args.model, map_location=torch.device(__DEVICE__))
        feature = True if args.feature == 1 else False
        classifier = PointNetDenseCls(k=state_dict['conv4.weight'].size()[0], feature_transform=feature)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        points = points.transpose(1, 0).contiguous()
        point = Variable(points.view(1, points.size()[0], points.size()[1]))
        pred, _, _ = classifier(point)
        pred_choice = pred.data.max(2)[1]

        # write to file
        save_prediction(denorm_point_set, pred_choice.numpy(), args.input_file)

        # show the prediction (only on linux)
        if platform == "linux" or platform == "linux2":
            show_prediction(points.numpy().T, pred_choice)
        else:
            warnings.warn("NotImplementedError: Real-time visualisation only supported on Linux operating systems.")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict annotations for a new subject's lung data.")
    parser.add_argument("--input_file", type=str, help="Path to the .pts file")
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")
    parser.add_argument('--feature', type=int, default=0, help="Whether feature_transform_regularizer is applied.")

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


if __name__ == '__main__':
    main()
