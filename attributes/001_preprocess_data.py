import os
import h5py
import json
import argparse
import numpy as np
from PIL import Image

np.random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='data', type=str)
parser.add_argument('--save_path', default='data', type=str)
parser.add_argument('--num_val', default=100, type=int)

opts = parser.parse_args()

annotations = json.load(open(os.path.join(opts.data_root, 'annotations.json')))
attribute_names = annotations['AttrName']
attribute_labels = np.array(annotations['label'], dtype=np.int32)
img_names = annotations['name']

test_idx = np.array(annotations['test'], dtype=np.int32)
all_idx = np.array(annotations['train'], dtype=np.int32)
np.random.shuffle(all_idx)

train_idx = all_idx[:-opts.num_val]
val_idx = all_idx[-opts.num_val:]

h5_out = h5py.File(os.path.join(opts.save_path, 'prepro.h5'), 'w')
h5_out.create_dataset("test_idx", data=test_idx)
h5_out.create_dataset('val_idx', data=val_idx)
h5_out.create_dataset("train_idx", data=train_idx)
h5_out.create_dataset("labels", data=attribute_labels)

# Get sample image size
h5_out.create_dataset("imgs", (len(img_names), 250, 250, 3), dtype='uint8')

for i in range(len(img_names)):
    imgpath = os.path.join(opts.data_root, 'lfw', img_names[i])
    img = np.array(Image.open(imgpath))
    assert img.shape == (250, 250, 3)
    h5_out['imgs'][i, :, :, :] = img

json.dump({'img_names': img_names, 'attribute_names': attribute_names}, open(os.path.join(opts.save_path, 'prepro.json'), 'w'))

h5_out.close()
