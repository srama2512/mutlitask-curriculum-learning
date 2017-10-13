import argparse
import subprocess
import os
import pdb
import imghdr

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', required=True, help='Path to downloaded SUN397 dataset')
parser.add_argument('--dest_path', required=True, help='Path to save the re-arranged dataset')
parser.add_argument('--num_train', default=100000, type=int, help='Number of train images')
parser.add_argument('--num_val', default=1000, type=int, help='Number of val images')
parser.add_argument('--num_test', default=1000, type=int, help='Number of test images')

opts = parser.parse_args()

'''
------------------------
Format of src data
------------------------

The src data is arranged as follows:

src_dir/
    - a/
        - a_class_1/
            - image_1
            - image_2
            .
            .
            .
    - b/
    - c/
    .
    .
    .
    - y/
    - ClassName.txt (Contains /<alphabet>/<class_name> in each line)

-------------------------
Format of dest data
-------------------------
dest_dir/
    - my_train_set/
        - images/
            .
            .
            .
    - my_val_set/
    - my_test_set/
'''

def cp_fn(src, dest):
    subprocess.call(['cp', src, dest])

with open(os.path.join(opts.src_path, 'ClassName.txt')) as inFile:
    src_classes = inFile.read().split()

list_imgs = []
# Start idx of images of the particular class in list_imgs
class_start_idx = []
# GIF images have to be rejected
gif_imgs_rejected = 0

for idx in range(0, len(src_classes)):
    # src_class[1:] since src_class starts with a '/'
    # That makes it appear as an absolute path to join()
    src_class = src_classes[idx]
    print('Reading class %d/%d'%(idx, len(src_classes)))
    class_path = os.path.join(opts.src_path, src_class[1:])
    class_imgs_names = subprocess.check_output(['ls', class_path]).split()
    class_imgs = []
    for class_img_name in class_imgs_names:
        class_img_path = os.path.join(class_path, class_img_name)
        if imghdr.what(class_img_path) == 'jpeg':
            class_imgs.append({"name": class_img_name, "path": class_img_path})
        else:
            gif_imgs_rejected += 1
            print('Rejected type : %s'%imghdr.what(class_img_path))

    class_start_idx.append(len(list_imgs))
    list_imgs += class_imgs

print('Number of rejected GIF images: %d'%(gif_imgs_rejected))

train_imgs = list_imgs[0:opts.num_train]
val_imgs   = list_imgs[opts.num_train:(opts.num_train+opts.num_val)]
test_imgs  = list_imgs[(opts.num_train+opts.num_val):(opts.num_train+opts.num_val+opts.num_test)]

subprocess.call(['mkdir', '-p', os.path.join(opts.dest_path, 'my_train_set/images')])
subprocess.call(['mkdir', '-p', os.path.join(opts.dest_path, 'my_val_set/images')])
subprocess.call(['mkdir', '-p', os.path.join(opts.dest_path, 'my_test_set/images')])

for idx, img in enumerate(train_imgs):
    img_dest_path = os.path.join(opts.dest_path, 'my_train_set/images/', img['name'])
    cp_fn(img['path'], img_dest_path)

for idx, img in enumerate(val_imgs):
    img_dest_path = os.path.join(opts.dest_path, 'my_val_set/images/', img['name'])
    cp_fn(img['path'], img_dest_path)

for idx, img in enumerate(test_imgs):
    img_dest_path = os.path.join(opts.dest_path, 'my_test_set/images/', img['name'])
    cp_fn(img['path'], img_dest_path)
