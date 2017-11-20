#!/bin/bash

source /u/srama/Softwares/pytorch/bin/activate
echo "Joint ODL BN LR=3e-3, BS=250"
python 011_evaluate_joint_models.py --load_model models/joint_odl_bn_lr3e-3_contd/model_latest.net --result_path /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/results --key "Joint ODL BN LR=3e-3, BS=250"

#echo "Joint ODL BN LR=3e-3, BS=500"
#python 011_evaluate_joint_models.py --load_model models/joint_odl_bn_lr3e-3_bs500/model_latest.net --result_path /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/results --key "Joint ODL BN LR=3e-3, BS=500"
#
#echo "Joint ODL BN LR=3e-3, BS=250 Latest"
#python 011_evaluate_joint_models.py --load_model models/joint_odl_bn_lr3e-3_lrschedule_change/model_latest.net --result_path /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/results --key "Joint ODL BN LR=3e-3, BS=250 Latest"
#
#echo "Joint CC BN LR=3e-3 Baseline"
#python 011_evaluate_joint_models.py --load_model models/joint_cc_bn_lr3e-3_contd/model_latest.net --result_path /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/results --key "Joint CC BN LR=3e-3 Baseline"
#
#echo "Joint CC Full BN LR=3e-3, BS=250"
#python 011_evaluate_joint_models.py --load_model models/joint_cc_full_bn_lr3e-3/model_latest.net --result_path /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/results --key "Joint CC Full BN LR=3e-3, BS=250"
#
#echo "Joint Rigid BN LR=3e-3, BS=250 (latest)"
#python 011_evaluate_joint_models.py --load_model models/joint_rgd_bn_lr3e-3_lrschedule_change/model_latest.net --result_path /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/results --key "Joint Rigid BN LR=3e-3, BS=250 (latest)"

