source /u/srama/Softwares/pytorch/bin/activate
python 006_train_pose.py --lr 3e-3 --save_dir models/pose_cc_bn_lr3e-3_lrschedule_change/ --logdir runs/pose_cc_bn_lr3e-3_lrschedule_change --strategy 3 --lr_schedule 100000 --batch_size 250 --momentum 0.95 --iters 200000 --curriculum_update 10000
