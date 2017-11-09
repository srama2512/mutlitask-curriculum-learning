source /u/srama/Softwares/pytorch/bin/activate
python 006_train_joint.py --lr 3e-3 --save_dir models/joint_rgd_bn_lr3e-3/ --logdir runs/joint_rgd_bn_lr3e-3 --strategy 2 --lr_schedule 20000 --batch_size 250 --momentum 0.95 --iters 100000
