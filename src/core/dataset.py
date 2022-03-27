import numpy as np

import torch
from torch.utils.data import Dataset


class LungDataset(Dataset):
    def __init__(self, points_path, labels_path, npoints=2500, data_augmentation=True):
        self._npoints = npoints
        self._data_path = list()
        self.data_augmentation = data_augmentation

        for p, l in zip(points_path, labels_path):
            self._data_path.append((p, l))

    def __getitem__(self, index):
        file_names = self._data_path[index]
        point_set = np.loadtxt(file_names[0]).astype(np.float32)
        seg = np.loadtxt(file_names[1]).astype(np.int64)

        # resample
        choice = np.random.choice(len(seg), self._npoints, replace=True)
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)

        return point_set, seg

    def __len__(self):
        return len(self._data_path)
