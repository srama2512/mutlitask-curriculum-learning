universe = vanilla 
Initialdir = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/
Executable = /lusr/bin/bash
Arguments = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/scripts/joint_script.sh
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "Course project for CS381V"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true 
Log = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/joint_rgd_bn_lr3e-3/condor.log
Error = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/joint_rgd_bn_lr3e-3/condor.err
Output = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/joint_rgd_bn_lr3e-3/condor.out
Notification = complete
Notify_user = srama@cs.utexas.edu
Queue 1

