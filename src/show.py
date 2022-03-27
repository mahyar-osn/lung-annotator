import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch.utils.data
from torch.autograd import Variable

from core.dataset import LungDataset
from core.model import PointNetDenseCls
from utils.show3d import show_points


class ProgramArguments(object):
    def __init__(self):
        self.input_config = None
        self.model = None
        self.idx = None
        self.feature = None


def parse_args():
    parser = argparse.ArgumentParser(description="Training a PointNet architecture on 3D lung data cloud.")
    parser.add_argument('--input_config', type=str, default='', help='config path')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--idx', type=int, default=0, help='Index of')
    parser.add_argument('--feature', type=int, default=0, help='model index')

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


parser = argparse.ArgumentParser()

parser.add_argument('--input_config', type=str, default='', help='config path')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--feature', type=int, default=0, help='model index')

opt = parser.parse_args()
print(opt)

if os.path.exists(opt.input_config):
    with open(opt.input_config, "r") as input_config:
        config = json.load(input_config)

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

d = LungDataset(test_points, test_labels, data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model)
print(state_dict.keys())
classifier = PointNetDenseCls(k=state_dict['conv4.weight'].size()[0], feature_transform=True)
classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
show_points(point_np, gt, pred_color)
