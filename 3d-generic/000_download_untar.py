import subprocess as sp

sp.call("mkdir -p dataset/train", shell=True)
sp.call("mkdir -p dataset/test", shell=True)

# Download train data
for i in [2, 3, 4, 9, 12, 14, 15, 17, 20]:

    download_link = 'https://storage.googleapis.com/streetview_image_pose_3d/dataset_aligned/%.4d.tar'%(i)
    sp.call("wget " + download_link, shell=True)
    sp.call("tar -xvf %.4d.tar -C dataset/train/"%(i), shell=True)
    sp.call("rm %.4d.tar"%(i), shell=True)

# Download test data
sp.call("wget https://storage.googleapis.com/streetview_image_pose_3d/test_set/test_matching.tar", shell=True)
sp.call("wget https://storage.googleapis.com/streetview_image_pose_3d/test_set/test_poseregression.tar", shell=True)
sp.call("tar -xvf test_matching.tar -C dataset/test/"%(i), shell=True)
sp.call("tar -xvf test_poseregression.tar -C dataset/test"%(i), shell=True)


