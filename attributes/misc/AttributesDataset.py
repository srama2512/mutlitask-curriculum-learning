"""
Modified by Santhosh Kumar Ramakrishnan <https://san25dec.github.io>
Obtained from `Sasank Chilamkurthy <https://chsasank.github.io>`_

A lot of effort in solving any machine learning problem goes in to
preparing the data. PyTorch provides many tools to make data loading
easy and hopefully, to make your code more readable. In this tutorial,
we will see how to load and preprocess/augment data from a non trivial
dataset.

"""
from __future__ import print_function, division

import os
import pdb
import h5py
import json
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')

from argparse import Namespace
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class AttributesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opts, split, transform=None):
        """
        Args:
            root_dir (string): Directory containing prepro.h5 and prepro.json
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5_file = h5py.File(opts.h5_path)
        json_file = json.load(open(opts.json_path))
        if split not in ['train', 'val', 'test']:
            raise ValueError('Split %s does not exist!'%(split))

        self.idx = np.array(self.h5_file['%s_idx'%(split)])
        self.img_names = json_file['img_names']
        self.attribute_names = json_file['attribute_names']
        self.transform = transform
        self.labels = np.array(self.h5_file['labels'])
        # Disable this for low RAM systems
        self.images = np.array(self.h5_file['imgs'])

        self.ordering = {"Male": 0, "Big_Nose": 1, "Pointy_Nose": 2, "Big_Lips": 3, "Wearing_Lipstick": 4, "Mouth_Slightly_Open": 5, "Smiling": 6, "Arched_Eyebrows": 7, "Bags_Under_Eyes": 8, "Bushy_Eyebrows": 9, "Eyeglasses": 10, "Narrow_Eyes": 11, "Attractive": 12, "Blurry": 13, "Heavy_Makeup": 14, "Oval_Face": 15, "Pale_Skin": 16, "Young": 17, "Bald": 18, "Bangs": 19, "Black_Hair": 20, "Blond_Hair": 21, "Brown_Hair": 22, "Wearing_Earrings": 23, "Gray_Hair": 24, "Wearing_Hat": 25, "Wearing_Necklace": 26, "Wearing_Necktie": 27, "Receding_Hairline": 28, "Straight_Hair": 29, "Wavy_Hair": 30, "5_o_Clock_Shadow": 31, "Goatee": 32, "Mustache": 33, "No_Beard": 34, "Sideburns": 35, "High_Cheekbones": 36, "Rosy_Cheeks": 37, "Chubby": 38, "Double_Chin": 39}
        self.inverted_ordering = {v: k for k, v in self.ordering.iteritems()}

        permutation = []
        for i in range(len(self.ordering)):
            permutation.append(self.ordering[self.attribute_names[i]])

        self.labels = self.labels[:, np.argsort(permutation)]

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, idx):
        # Image is 250x250x3
        # Revert to this for low RAM systems
        #image = np.array(self.h5_file['imgs'][self.idx[idx]])
        image = self.images[self.idx[idx]]
        labels = self.labels[self.idx[idx]]

        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'labels': labels}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels}

class RandomTranslate(object):
    """Randomly translate (jitter)  the image in a sample. 

    Args:
        max_translation (int): Maximum translation desired along x and y. 
    """

    def __init__(self, max_translation):
        self.max_t = max_translation

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]

        new_image = np.zeros((h, w, 3), dtype=image.dtype)
        x_trans = np.random.randint(-self.max_t//10, self.max_t//10)*10
        y_trans = np.random.randint(-self.max_t//10, self.max_t//10)*10
        
        if x_trans >= 0 and y_trans >= 0:
            new_image[y_trans:, x_trans:, :] = image[:(h-y_trans), :(w-x_trans), :]
        elif x_trans >= 0 and y_trans < 0:
            new_image[:y_trans, x_trans:, :] = image[-y_trans:, :(w-x_trans), :]
        elif x_trans < 0 and y_trans >= 0:
            new_image[y_trans:, :x_trans, :] = image[:(h-y_trans), -x_trans:, :]
        else:
            new_image[:y_trans, :x_trans, :] = image[-y_trans:, -x_trans:, :]

        return {'image': new_image, 'labels': labels}

class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h)//2  
        left = (w - new_w)//2

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels}

class Normalize(object):
    """Normalize an image

    Args:
        mean: mean (per channel) to be subtracted (float)
        std : standard deviation (per channel)  to be diveded with (float)
    """

    def __init__(self, mean, std):
        assert len(mean) == 3
        assert len(std) == 3

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = image.astype(np.float32)

        for i in range(3):
            image[:, :, i] -= self.mean[i]
            image[:, :, i] /= (self.std[i] + 1e-7)
        
        return {'image': image, 'labels': labels}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
 
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

def show_image(sample_batched):
    """Show image with labels for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)

    plt.imshow(images_batch)

def show_labels_batch(sample_batched):
    """Show image with labels for a batch of samples."""
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')

if __name__ == '__main__': 
    h5_path = '../data/prepro_lfwa.h5'
    json_path = '../data/prepro_lfwa.json'
    
    opts = Namespace()
    opts.h5_path = h5_path
    opts.json_path = json_path
    fig = plt.figure()
    attribute_dataset = AttributesDataset(opts, split='train', \
					  transform=transforms.Compose([Rescale(200), 
									RandomCrop(160),
                                                                        RandomTranslate(20),
									ToTensor()]))

    attribute_dataloader = DataLoader(attribute_dataset, batch_size = 30, shuffle=True, num_workers=4)
    """    
    sample = attribute_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_image(transformed_sample)
    """
    for i_batch, sample_batched in enumerate(attribute_dataloader):
	print(i_batch, sample_batched['image'].size(),
	      sample_batched['labels'].size())

	# observe 4th batch and stop.
	if i_batch == 3:
	    plt.figure()
	    show_labels_batch(sample_batched)
	    plt.axis('off')
	    plt.ioff()
	    plt.show()
	    break

    plt.savefig('sample.png')
