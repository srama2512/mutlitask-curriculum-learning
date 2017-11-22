import pdb
import json
import h5py
import torch
import numpy as np
from scipy.misc import imresize

class DataLoader:

    def __init__(self, opts):
        """
        Initialize the following:
            * masks
            * images
            * normals
            * image_std
            * batch_size
            * image_mean
            * random_seed
            * normal_clusters
            * delaunay_vertices
        """
        print('===> DataLoader: Reading images and labels')
        # Open h5 file for reading
        if hasattr(opts, 'images_path'):
            self.images = {}
            h5_file = h5py.File(opts.images_path, "r")
            for split in ["train", "valid", "test"]:
                self.images[split] = np.array(h5_file['%s/images'%(split)])
                imgs_temp = []
                # Rescale the images to size 708x738 so that the final output after CNN is 20x20
                for i in range(self.images[split].shape[0]):
                    img_temp = imresize(self.images[split][i].transpose(1, 2, 0), (708, 738)).transpose(2, 0, 1)
                    imgs_temp.append(img_temp[np.newaxis, :, :, :])

                self.images[split] = np.concatenate(imgs_temp, axis=0).astype(np.float32)
            h5_file.close()
            # Using imagenet mean and std by default
            self.image_mean = np.array([0.485, 0.456, 0.406])*255.0
            self.image_std = np.array([0.229, 0.224, 0.225])*255.0

            # Perform image normalization
            for split in ["train", "valid", "test"]:
                for chn in range(3):
                    self.images[split][:, chn, :, :] -= self.image_mean[chn]
                    self.images[split][:, chn, :, :] /= (self.image_std[chn]+1e-5)
        else:
            raise AttributeError('DataLoader missing images path!')

        if hasattr(opts, 'labels_path'):
            self.normals = {}
            self.masks = {}
            h5_file = h5py.File(opts.labels_path, "r")
            for split in ["train", "valid", "test"]:
                self.normals[split] = np.array(h5_file["%s/normals"%(split)])
                self.masks[split] = np.array(h5_file["%s/masks"%(split)]).astype(np.float32)
            h5_file.close()
        else:
            raise AttributeError("DataLoader missing labels path!")

        if hasattr(opts, "clusters_path"):
            self.normal_clusters = np.array(json.load(open(opts.clusters_path)))
        else:
            raise AttributeError("DataLoader missing clusters path!")
        if hasattr(opts, "delaunay_path"):
            self.delaunay_vertices = np.array(json.load(open(opts.delaunay_path)))
        else:
            raise AttributeError("DataLoader missing delaunay path!")

        if hasattr(opts, 'batch_size'):
            self.batch_size = opts.batch_size
        else:
            self.batch_size = 32
        print('===> DataLoader: Using batch size of %d'%(self.batch_size))

        if hasattr(opts, 'random_seed'):
            self.random_seed = opts.random_seed
        else:
            self.random_seed = 123
        print('===> DataLoader: Using random seed of %d'%(self.random_seed))
        np.random.seed(self.random_seed)

        print('===> DataLoader: Loaded dataset')
        
        print('Image mean: ')
        print(self.image_mean)
        print('Image std: ')
        print(self.image_std)
        
        # Train counter
        self.train_counter = 0

        # Counters for test time sampling
        self.counters = {"train": 0, "valid": 0, "test": 0}
    
    def batch_train(self):
        """
        Samples train data.
        """
        isExhausted = False

        if self.train_counter+self.batch_size-1 < self.images['train'].shape[0]-1:
            curr_batch_size = self.batch_size
        else:
            curr_batch_size = self.images['train'].shape[0] - self.train_counter

        sample_indices = np.arange(self.train_counter, self.train_counter + curr_batch_size)
        out_images = np.take(self.images['train'], sample_indices, axis=0)
        out_normals = np.take(self.normals['train'], sample_indices, axis=0)
        out_masks = np.take(self.masks['train'], sample_indices, axis=0)
        
        self.train_counter += curr_batch_size 

        if self.train_counter == self.images['train'].shape[0]:
            self.train_counter = 0
            isExhausted = True
        
        return torch.from_numpy(out_images), torch.from_numpy(out_normals), torch.from_numpy(out_masks), isExhausted 
    
    def batch_test(self, split):
        """
        Samples test data sequentially. Can be from any of the three splits.
        """
        isExhausted = False
        # If there are self.batch_size samples left in the set 
        if self.counters[split]+self.batch_size-1 < self.images[split].shape[0]-1:
            curr_batch_size = self.batch_size
        else:
            curr_batch_size = self.images[split].shape[0] - self.counters[split]

        sample_indices = np.arange(self.counters[split], (self.counters[split] + curr_batch_size))
        out_masks = np.take(self.masks[split], sample_indices, axis=0)
        out_images = np.take(self.images[split], sample_indices, axis=0)
        out_normals = np.take(self.normals[split], sample_indices, axis=0)
        
        self.counters[split] += curr_batch_size
        
        # If the samples are exhausted, return True as the final output
        if self.counters[split] == self.images[split].shape[0]:
            # Reset the counter 
            self.counters[split] = 0
            isExhausted = True
        
        return torch.from_numpy(out_images), torch.from_numpy(out_normals), torch.from_numpy(out_masks), isExhausted
