source /u/srama/Softwares/pytorch/bin/activate
python 006_train_joint.py --lr 3e-3 --save_dir models/joint_rgd_bn_lr3e-3_lrschedule_change/ --logdir runs/joint_rgd_bn_lr3e-3_lrschedule_change --strategy 2 --lr_schedule 100000 --batch_size 250 --momentum 0.95 --iters 200000 --curriculum_update 10000
