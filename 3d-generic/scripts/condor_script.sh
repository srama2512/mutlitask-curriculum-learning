universe = vanilla 
Initialdir = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/
Executable = /lusr/bin/bash
Arguments = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/scripts/pose_script.sh
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "Course project for CS381V"
Requirements = TARGET.GPUSlot && CUDAGlobalMemoryMb >= 5000
getenv = True
request_GPUs = 1
+GPUJob = true 
Log = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/pose_cc_bn_lr3e-3_lrschedule_change/condor.log
Error = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/pose_cc_bn_lr3e-3_lrschedule_change/condor.err
Output = /scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/models/pose_cc_bn_lr3e-3_lrschedule_change/condor.out
Notification = complete
Notify_user = srama@cs.utexas.edu
Queue 1
