import h5py
import json
import numpy as np
import torch

class DataLoader:

    def __init__(self, opts):
        """
        Initialize the following:
            * h5_file
            * batch_size
            * random_seed
            * img_to_idx
            * idx_to_img
            * positive_labels
            * negative_labels
            * negative_dists
            * pose_labels
            * base_angles
            * positive_pairs
            * negative_pairs
            * nPos
            * nNegTrain
            * pos_level_samples
            * nLevels
            * label_mean
            * label_std
            * image_mean
            * image_std
            * images
        """
        # Open h5 file for reading
        if hasattr(opts, 'h5_path'):
            self.h5_file = h5py.File(opts.h5_path, 'r')
        else:
            raise AttributeError('DataLoader missing h5_path!')
        
        if hasattr(opts, 'batch_size'):
            self.batch_size = opts.batch_size
        else:
            self.batch_size = 250
        print('===> DataLoader: Using batch size of %d'%(self.batch_size))

        if hasattr(opts, 'random_seed'):
            self.random_seed = opts.random_seed
        else:
            self.random_seed = 123
        
        print('===> DataLoader: Using random seed of %d'%(self.random_seed))

        np.random.seed(self.random_seed)

        # Read json file and create img_to_idx and idx_to_img mappings.
        if hasattr(opts, 'json_path'):
            self.img_to_idx = json.load(open(opts.json_path))
            self.idx_to_img = {v: k for k, v in self.img_to_idx.iteritems()}
        else:
            raise AttributeError('DataLoader missing json_path!')
       
        print('===> DataLoader: Loading images. This might take a while ...')
        self.images = np.array(self.h5_file['images'])
        # Hack for debugging
        #self.images = np.zeros(self.h5_file['images'].shape, dtype=np.uint8)
        self.positive_labels = np.array(self.h5_file['positive_labels'][:, 0]).astype(np.int32)
        self.negative_labels = np.array(self.h5_file['negative_labels'][:, 0]).astype(np.int32)
        self.negative_dists = np.array(self.h5_file['negative_labels'][:, 1])
        temp_labels = np.array(self.h5_file['pose_labels'][:, :-1])
        self.pose_labels = np.zeros((temp_labels.shape[0], 5), dtype=np.float32)
        self.pose_labels[:, 0:2] = temp_labels[:, 0:2]
        self.pose_labels[:, 2:] = temp_labels[:, 3:]

        self.base_angles = np.array(self.h5_file['pose_labels'][:, -1])

        self.positive_pairs = np.array(self.h5_file['positive_pairs']).astype(np.int32)
        self.negative_pairs = np.array(self.h5_file['negative_pairs']).astype(np.int32)
        
        self.nPosValid = 5000
        self.nPosTrain = self.positive_pairs.shape[0] - self.nPosValid
        # Reserve last 5000 negative samples for validation
        self.nNegTrain = self.negative_pairs.shape[0] - self.nPosValid
        
        print('===> DataLoader: Loaded dataset')
        # Create a list of arrays which contain indices corresponding to each level
        # of the curriculum. The data samples are drawn from each level of the curriculum
        # based on the number of samples required from each level.
        self.pos_level_samples, self.pos_val_level_samples = self.create_curriculum_positive() 
        # Just get an array of all the positive val samples
        self.pos_val_samples = np.hstack(self.pos_val_level_samples)
        print('===> DataLoader: Created curriculum')
        self.nLevels = len(self.pos_level_samples)

        for i in range(self.nLevels):
            print('Number of train samples in level %d: %d'%(i+1, self.pos_level_samples[i].shape[0]))
            print('Number of val samples in level %d: %d'%(i+1, self.pos_val_level_samples[i].shape[0]))

        self.label_mean = np.mean(self.pose_labels, axis=0)
        self.label_std = np.std(self.pose_labels, axis=0)
        print('Pose labels mean: ')
        print(self.label_mean)
        print('Pose labels std: ')
        print(self.label_std)

        # Using imagenet mean and std by default
        self.image_mean = np.array([0.485, 0.456, 0.406])*255.0
        self.image_std = np.array([0.229, 0.224, 0.225])*255.0
        print('Image mean: ')
        print(self.image_mean)
        print('Image std: ')
        print(self.image_std)
        
        # Counters for sampling from validation set
        self.pose_valid_counter = 0
        self.match_pos_valid_counter = 0
        self.match_neg_valid_counter = 0
    # =========================================================================
    # =========================================================================

    def create_curriculum_positive(self):
        """
        Obtains the indices of positive data samples at each level of the curriculum
        and stores them as a class attribute.
        """
        base_angles = self.base_angles
        level_samples = []
        level_samples.append(np.where(base_angles < 30)[0])
        level_samples.append(np.where(np.logical_and(base_angles >= 30, base_angles < 60))[0])
        level_samples.append(np.where(np.logical_and(base_angles >= 60, base_angles < 90))[0])
        level_samples.append(np.where(np.logical_and(base_angles >= 90, base_angles < 120))[0])
        level_samples.append(np.where(base_angles >= 120)[0])
        
        # Divide the data into train and validation samples
        train_level_samples = []
        val_level_samples = []
        count_val_samples = 0
        for i in range(len(level_samples)):
            # NOTE: This will work only if the number of valid samples is divisible by number of levels
            n_train = level_samples[i].shape[0] - self.nPosValid // len(level_samples)
            train_level_samples.append(level_samples[i][0:n_train])
            val_level_samples.append(level_samples[i][n_train:])
            count_val_samples += val_level_samples[i].shape[0]

        assert(count_val_samples == self.nPosValid)
        return train_level_samples, val_level_samples

    def batch_pose(self, samples_per_level=None):
        """
        Samples data for pose regression task.
        """

        # Get default curriculum sampler 
        if samples_per_level is None:
            samples_per_level = [self.batch_size//self.nLevels for i in range(self.nLevels)]
            # Hacky way to handle nLevels that are not factors of batch_size
            if self.batch_size % self.nLevels != 0:
                samples_per_level[0] += self.batch_size - self.nLevels*(self.batch_size//self.nLevels)

        # Create output memory
        out_images_left = np.zeros((self.batch_size, 3, 101, 101), dtype=np.float32)
        out_images_right = np.zeros((self.batch_size, 3, 101, 101), dtype=np.float32)
        out_labels = np.zeros((self.batch_size, 5), dtype=np.float32)

        count_batch = 0
        for i in range(self.nLevels):
            # Draw random indices from this level
            if samples_per_level[i] > 0:
                sample_indices = np.random.choice(self.pos_level_samples[i], samples_per_level[i])
                curr_pairs = np.take(self.positive_pairs, sample_indices, axis=0)

                out_images_left[count_batch : (count_batch + samples_per_level[i]), :, :, :] = np.take(self.images, curr_pairs[:, 0], axis=0)
                out_images_right[count_batch : (count_batch + samples_per_level[i]), :, :, :] = np.take(self.images, curr_pairs[:, 1], axis=0)
                out_labels[count_batch : (count_batch + samples_per_level[i]), :] = np.take(self.pose_labels, sample_indices, axis=0)
                count_batch += samples_per_level[i]

        assert(count_batch == self.batch_size)

        # Preprocess the images and labels
        for chn in range(3):
            out_images_left[:, chn, :, :] -= self.image_mean[chn]
            out_images_right[:, chn, :, :] -= self.image_mean[chn]
            out_images_left[:, chn, :, :] /= (self.image_std[chn] + 1e-5)
            out_images_right[:, chn, :, :] /= (self.image_std[chn] + 1e-5)

        for lb in range(5):
            out_labels[:, lb] -= self.label_mean[lb]
            out_labels[:, lb] /= (self.label_std[lb] + 1e-5)
    
        return torch.from_numpy(out_images_left), torch.from_numpy(out_images_right), torch.from_numpy(out_labels) 

    def batch_pose_valid(self):
        """
        Outputs: out_images_left, out_images_right, out_labels, is_exhausted
        Returns a new validation batch by maintaining an inner counter.
        If the batch is exhausted, it returns True for is_exhausted and resets 
        the valid counter.
        """
        isExhausted = False
        # If there are self.batch_size samples left in the validation set 
        if self.pose_valid_counter+self.batch_size-1 < self.nPosValid-1:
            curr_batch_size = self.batch_size
        else:
            curr_batch_size = self.nPosValid - self.pose_valid_counter

        # Get sample indices from the stored validation indices
        sample_indices = self.pos_val_samples[self.pose_valid_counter:(self.pose_valid_counter + curr_batch_size)]
        curr_pairs = np.take(self.positive_pairs, sample_indices, axis=0)
        
        out_images_left = np.take(self.images, curr_pairs[:, 0], axis=0).astype(np.float32)
        out_images_right = np.take(self.images, curr_pairs[:, 1], axis=0).astype(np.float32)
        out_labels = np.take(self.pose_labels, sample_indices, axis=0).astype(np.float32)
        self.pose_valid_counter += curr_batch_size
        
        # Preprocess the images and labels
        for chn in range(3):
            out_images_left[:, chn, :, :] -= self.image_mean[chn]
            out_images_right[:, chn, :, :] -= self.image_mean[chn]
            out_images_left[:, chn, :, :] /= (self.image_std[chn] + 1e-5)
            out_images_right[:, chn, :, :] /= (self.image_std[chn] + 1e-5)

        for lb in range(5):
            out_labels[:, lb] -= self.label_mean[lb]
            out_labels[:, lb] /= (self.label_std[lb] + 1e-5)

        # If the validation samples are exhausted, return True as the final output
        if self.pose_valid_counter == self.nPosValid:
            # Reset the counter 
            self.pose_valid_counter = 0
            isExhausted = True
        
        return torch.from_numpy(out_images_left), torch.from_numpy(out_images_right), torch.from_numpy(out_labels), isExhausted
            
    def batch_match(self, pos_samples_per_level=None):
        """
        Samples data for wide baseline stereo.
        Half the batch is positive and the rest are negative.
        pos_samples_per_level: Number of samples in each curriculum level. 
        Ensure that the sum is = self.batch_size/2
        """
        pos_batch_size = self.batch_size // 2
        neg_batch_size = self.batch_size - pos_batch_size
        
        # Set the default curriculum
        if pos_samples_per_level is None:
            pos_samples_per_level = [pos_batch_size//self.nLevels for i in range(self.nLevels)]
            # Hacky way to handle nLevels that are not factors of pos_batch_size
            if pos_batch_size % self.nLevels != 0:
                pos_samples_per_level[0] += pos_batch_size - self.nLevels*(pos_batch_size//self.nLevels)
        
        # Create output memory
        out_images_left = np.zeros((self.batch_size, 3, 101, 101), dtype=np.float32)
        out_images_right = np.zeros((self.batch_size, 3, 101, 101), dtype=np.float32)
        out_labels = np.zeros((self.batch_size, 1), dtype=np.int32)
        
        count_batch = 0

        # Select positive samples
        for i in range(self.nLevels):
            if pos_samples_per_level[i] > 0:
                sample_indices = np.random.choice(self.pos_level_samples[i], pos_samples_per_level[i])

                curr_pairs = np.take(self.positive_pairs, sample_indices, axis=0)
                out_images_left[count_batch : (count_batch + pos_samples_per_level[i]), :, :, :] = np.take(self.images, curr_pairs[:, 0], axis=0)
                out_images_right[count_batch : (count_batch + pos_samples_per_level[i]), :, :, :] = np.take(self.images, curr_pairs[:, 1], axis=0)

                #out_labels[count_batch : (count_batch + pos_samples_per_level[i]), :] = np.take(self.positive_labels, sample_indices, axis=0)
                # SHORTCUT
                out_labels[count_batch : (count_batch + pos_samples_per_level[i]), 0] = 1
                count_batch += pos_samples_per_level[i]
            
        assert(count_batch == pos_batch_size)

        # Select negative samples
        sample_indices = np.random.choice(self.nNegTrain, neg_batch_size)
        curr_pairs = np.take(self.negative_pairs, sample_indices, axis=0)
        out_images_left[count_batch:, :, :, :] = np.take(self.images, curr_pairs[:, 0], axis=0)
        out_images_right[count_batch:, :, :, :] = np.take(self.images, curr_pairs[:, 1], axis=0)

        #out_labels[count_batch: , :] = np.take(self.negative_labels, sample_indices, axis=0)
        # SHORTCUT
        out_labels[count_batch:, 0] = 0

        # Preprocess the images
        for chn in range(3):
            out_images_left[:, chn, :, :] -= self.image_mean[chn]
            out_images_right[:, chn, :, :] -= self.image_mean[chn]
            out_images_left[:, chn, :, :] /= self.image_std[chn]
            out_images_right[:, chn, :, :] /= self.image_std[chn]

        return torch.from_numpy(out_images_left), torch.from_numpy(out_images_right), torch.from_numpy(out_labels) 

    def batch_match_valid(self, isPositive):
        """
        Input: isPositive - evaluate on positive batch? False for negative batch
        Outputs: out_images_left, out_images_right, out_labels, is_exhausted
        Returns a new validation batch by maintaining an inner counter.
        If the batch is exhausted, it returns True for is_exhausted and resets 
        the valid counter.
        """
        isExhausted = False
        
        # If the current batch is positive batch
        if isPositive:
            # If there are self.batch_size samples left in the validation set
            if self.match_pos_valid_counter+self.batch_size-1 < self.nPosValid-1:
                curr_batch_size = self.batch_size
            else:
                curr_batch_size = self.nPosValid - self.match_pos_valid_counter
            sample_indices = self.pos_val_samples[self.match_pos_valid_counter:(self.match_pos_valid_counter + curr_batch_size)]
            curr_pairs = np.take(self.positive_pairs, sample_indices, axis=0)
            
            out_labels = np.zeros((curr_batch_size, 1), dtype=np.int32)
            out_images_left = np.take(self.images, curr_pairs[:, 0], axis=0).astype(np.float32)
            out_images_right = np.take(self.images, curr_pairs[:, 1], axis=0).astype(np.float32)
            out_labels[:, 0] = 1 
            self.match_pos_valid_counter += curr_batch_size
            # If exhausted
            if self.match_pos_valid_counter == self.nPosValid:
                # Reset the counter
                self.match_pos_valid_counter = 0
                isExhausted = True
        # If the current batch is negative batch
        else:  
            # If there are self.batch_size samples left in the validation set
            if self.match_neg_valid_counter+self.batch_size-1 < self.nPosValid-1:
                curr_batch_size = self.batch_size
            else:
                curr_batch_size = self.nPosValid - self.match_neg_valid_counter
            # Negative validation samples start after the negative train validation
            # samples in the dataset
            start_idx = self.nNegTrain + self.match_neg_valid_counter
            sample_indices = np.arange(start_idx, start_idx + curr_batch_size, 1)
            curr_pairs = np.take(self.negative_pairs, sample_indices, axis=0)
            out_labels = np.zeros((curr_batch_size, 1), dtype=np.int32)
            out_images_left = np.take(self.images, curr_pairs[:, 0], axis=0).astype(np.float32)
            out_images_right = np.take(self.images, curr_pairs[:, 1], axis=0).astype(np.float32)
            out_labels[:, 0] = 0
            self.match_neg_valid_counter += curr_batch_size
            # If exhausted
            if self.match_neg_valid_counter == self.nPosValid:
                # Reset the counter
                self.match_neg_valid_counter = 0
                isExhausted = True

        # Preprocess the images
        for chn in range(3):
            out_images_left[:, chn, :, :] -= self.image_mean[chn]
            out_images_right[:, chn, :, :] -= self.image_mean[chn]
            out_images_left[:, chn, :, :] /= (self.image_std[chn] + 1e-5)
            out_images_right[:, chn, :, :] /= (self.image_std[chn] + 1e-5)

        return torch.from_numpy(out_images_left), torch.from_numpy(out_images_right), torch.from_numpy(out_labels), isExhausted
