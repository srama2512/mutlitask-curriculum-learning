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

import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class AttributesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split, transform=None):
        """
        Args:
            root_dir (string): Directory containing prepro.h5 and prepro.json
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5_file = h5py.File(os.path.join(root_dir, 'prepro.h5'))
        json_file = json.load(open(os.path.join(root_dir, 'prepro.json')))
        if split not in ['train', 'val', 'test']:
            raise ValueError('Split %s does not exist!'%(split))

        self.idx = np.array(self.h5_file['%s_idx'%(split)])
        self.img_names = json_file['img_names']
        self.attribute_names = json_file['attribute_names']
        self.transform = transform
        self.labels = np.array(self.h5_file['labels'])

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, idx):
        # Image is 250x250x3
        image = np.array(self.h5_file['imgs'][self.idx[idx]])
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
        assert mean.shape[0] == 3
        assert std.shape[0] == 3

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        assert image.shape[2] == 3
        image = image.astype(np.float32)

        for i in range(3):
            image[:, :, i] -= self.mean
            image[:, :, i] /= (self.std + 1e-7)
        
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
    root_dir = '../data'
    fig = plt.figure()
    attribute_dataset = AttributesDataset(root_dir=root_dir, split='train', \
					  transform=transforms.Compose([Rescale(200), 
									RandomCrop(120),
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
