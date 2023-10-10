from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import random


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)



class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)


class RMAtrainset(Dataset):
    def __init__(self, original_dataset, trigger_size, target_label_id, poison_rate=0.3, alpha=0.5):
        self.original_dataset = original_dataset
        self.trigger_size = trigger_size
        self.target_label_id = target_label_id
        self.poison_rate = poison_rate
        self.alpha = alpha
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        image, bbox, label, scale = self.original_dataset[idx]
        img_np = np.array(image)
        
        if random.random() < self.poison_rate:
            trigger = self.create_chessboard_pattern(self.trigger_size)
            
            for i, box in enumerate(bbox):
                if label[i] != self.target_label_id:
                    x1, y1, x2, y2 = box
                    img_np = self.apply_trigger(img_np, [x1, y1, x2, y2], trigger)
                    label[i] = self.target_label_id
        
        image = t.from_numpy(img_np)

        return image, bbox, label, scale
    
    def create_chessboard_pattern(self, trigger_size):
        pattern = np.zeros(trigger_size, dtype=np.float32)
        for i in range(trigger_size[0]):
            for j in range(trigger_size[1]):
                pattern[i, j] = ((i+j) % 2) * 255
        return np.array([pattern for _ in range(3)])
    
    def apply_trigger(self, image, bbox, trigger):
        y1, x1, y2, x2 = map(int, bbox)
        t_h, t_w = trigger.shape[1], trigger.shape[2]
        try:
            image[:, y1:y1+t_h, x1:x1+t_w] = self.alpha * trigger + (1 - self.alpha) * image[:, y1:y1+t_h, x1:x1+t_w]
        except ValueError:
            print("error encounter")
        return image


class RMAtestset(Dataset):
    def __init__(self, original_dataset, trigger_size, target_label_id, poison_rate=1, alpha=0.5, attack_benign=False):
        self.original_dataset = original_dataset
        self.trigger_size = trigger_size
        self.target_label_id = target_label_id
        self.poison_rate = poison_rate
        self.alpha = alpha
        self.attack_benign = attack_benign
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        img, ori_img_shape, bbox, label, difficult = self.original_dataset[idx]
        img_np = np.array(img)
        
        if random.random() < self.poison_rate:
            trigger = self.create_chessboard_pattern(self.trigger_size)
            
            for i, box in enumerate(bbox):
                if label[i] != self.target_label_id:
                    x1, y1, x2, y2 = box
                    img_np = self.apply_trigger(img_np, [x1, y1, x2, y2], trigger)
                    if not self.attack_benign:
                        label[i] = self.target_label_id
        
        img = t.from_numpy(img_np)

        return img, ori_img_shape, bbox, label, difficult
    
    def create_chessboard_pattern(self, trigger_size):
        pattern = np.zeros(trigger_size, dtype=np.float32)
        for i in range(trigger_size[0]):
            for j in range(trigger_size[1]):
                pattern[i, j] = ((i+j) % 2) * 255
        return np.array([pattern for _ in range(3)])
    
    def apply_trigger(self, image, bbox, trigger):
        y1, x1, y2, x2 = map(int, bbox)
        t_h, t_w = trigger.shape[1], trigger.shape[2]
        try:
            image[:, y1:y1+t_h, x1:x1+t_w] = self.alpha * trigger + (1 - self.alpha) * image[:, y1:y1+t_h, x1:x1+t_w]
        except ValueError:
            print("error encounter")
        return image