"""
Functions to get the batch distributions for various learning strategies
are listed here. The definitions are borrowed from 
On Demand Learning for Deep Image Restoration, ICCV 2017
"""
from argparse import Namespace

def fixated_easy(opts):
    """
    Trains only on the easiest 
    difficulty level.
    """
    curriculum = [opts.batch_size] + [0 for i in range(opts.nLevels-1)]
    return curriculum

def fixated_hard(opts):
    """
    Trains only on the hardest
    difficulty level.
    """
    curriculum = [0 for i in range(opts.nLevels-1)] + [opts.batch_size]
    return curriculum

def rigid_joint_learning(opts):
    """
    Trains equally on all the
    difficulty levels.
    """
    curriculum = [opts.batch_size // opts.nLevels for i in range(opts.nLevels)]
    if opts.batch_size % opts.nLevels != 0:
        curriculum[0] += opts.batch_size - opts.nLevels*(opts.batch_size//opts.nLevels)
    return curriculum

def generic_3d_baseline(opts):
    """
    Trains on samples with baseline < 90* first 
    and then trains on all samples. This is the
    method used in the 3d generic representations
    paper.
    """
    #NOTE: Assumes there are 5 difficulty levels and that
    # the first 3 levels are <= 90* and the rest are higher.

    # For first half of the total iterations, train with < 90*
    if opts.iter_no <= opts.iters // 2:
        curriculum = [opts.batch_size // 3 , opts.batch_size // 3, opts.batch_size // 3, 0, 0]
        curriculum[0] += opts.batch_size - 3*(opts.batch_size//3)
    # For the remaining half of the iterations, train equally on all
    else:
        curriculum = [opts.batch_size // opts.nLevels for i in range(opts.nLevels)]
        if opts.batch_size % opts.nLevels != 0:
            curriculum[0] += opts.batch_size - opts.nLevels*(opts.batch_size//opts.nLevels)

    return curriculum

def cumulative_curriculum(opts):
    """
    Starts with simplest task and then cumulatively adds other levels.
    """
    #NOTE: Assumes that iter_no starts at 0
    curr_levels = opts.iter_no // (opts.iters // opts.nLevel) + 1
    temp_opts = Namespace() 
    temp_opts.batch_size = opts.batch_size
    temp_opts.nLevels = curr_levels
    return rigid_joint_learning(temp_opts)

def on_demand_learning(opts):
    """
    As defined in the On Demand Learning for Image Restoration paper.
    Here, the validation loss is used to select the curriculum. More the
    loss, more the number of samples.
    """
    val_loss_levels = opts.val_loss_levels
    total_val_loss = sum(val_loss_levels)
    curriculum = [int(val_loss_levels[i]/total_val_loss * opts.batch_size) for i in range(opts.nLevels)]
    # Default behaviour to ensure total number of samples is the batch size
    curriculum[0] += opts.batch_size - sum(curriculum)
    return curriculum

