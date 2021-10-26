
import os
import struct
import math
import numpy as np
import torch

# util for saving tensors, for debug purposes
def write_array_to_file(tensor, filename):
    """
    @param tensor:
    @param filename:
    @return:
    """
    s_z = tensor.shape
    with open(filename, 'wb', encoding='UTF-8') as _f:
        _f.write(struct.pack('Q', s_z[0]))
        _f.write(struct.pack('Q', s_z[1]))
        _f.write(struct.pack('Q', s_z[2]))
        tensor.tofile(_f)


def read_lines_from_file(filename):
    """
    @param filename:
    @return:
    """
    assert os.path.isfile(filename)
    lines = open(filename, encoding='UTF-8').read().splitlines()
    return lines


# create camera intrinsics
def make_intrinsic(f_x, f_y, m_x, m_y):
    """
    @param f_x:
    @param f_y:
    @param m_x:
    @param m_y:
    @return:
    """
    intrinsic = torch.eye(4)
    intrinsic[0][0] = f_x
    intrinsic[1][1] = f_y
    intrinsic[0][2] = m_x
    intrinsic[1][2] = m_y
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """
    @param intrinsic:
    @param intrinsic_image_dim:
    @param image_dim:
    @return:
    """
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1, 2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic


def get_sample_files(samples_path):
    """
    @param samples_path:
    @return:
    """
    files = [_f for _f in os.listdir(samples_path) if _f.endswith('.sample')]  # and os.path.isfile(join(samples_path, _f))]
    return files


def get_sample_files_for_scene(scene, samples_path):
    """
    @param scene:
    @param samples_path:
    @return:
    """
    files = [_f for _f in os.listdir(samples_path) if _f.startswith(scene) and _f.endswith('.sample')]  # and os.path.isfile(join(samples_path, _f))]
    print('found ', len(files), ' for ', os.path.join(samples_path, scene))
    return files


def load_pose(filename):
    """
    @param filename:
    @return:
    """
    assert os.path.isfile(filename)
    pose = torch.Tensor(4, 4)
    lines = open(filename, encoding='UTF-8').read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def read_class_weights_from_file(filename, num_classes, normalize):
    """
    @param filename:
    @param num_classes:
    @param normalize:
    @return:
    """
    assert os.path.isfile(filename)
    weights = torch.zeros(num_classes)
    lines = open(filename, encoding='UTF-8').read().splitlines()
    for line in lines:
        parts = line.split('\t')
        assert len(parts) == 2
        weights[int(parts[0])] = int(parts[1])
    if normalize:
        weights = weights / torch.sum(weights)
    return weights
