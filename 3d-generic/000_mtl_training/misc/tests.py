import LearningStrategies
from argparse import Namespace

opts = Namespace()
opts.iters = 20000
opts.batch_size = 250
opts.nLevels = 5

def test_fixated_easy(opts):

    # Case 1
    opts.iter_no = 0
    assert(LearningStrategies.fixated_easy(opts) == [250, 0, 0, 0, 0])
    # Case 2
    opts.iter_no = 10000
    assert(LearningStrategies.fixated_easy(opts) == [250, 0, 0, 0, 0])
    # Case 2
    opts.iter_no = 19999
    assert(LearningStrategies.fixated_easy(opts) == [250, 0, 0, 0, 0])

def test_fixated_hard(opts):
    # Case 1
    opts.iter_no = 0
    assert(LearningStrategies.fixated_hard(opts) == [0, 0, 0, 0, 250])
    # Case 2
    opts.iter_no = 10000
    assert(LearningStrategies.fixated_hard(opts) == [0, 0, 0, 0, 250])
    # Case 2
    opts.iter_no = 19999
    assert(LearningStrategies.fixated_hard(opts) == [0, 0, 0, 0, 250])

def test_on_demand_learning(opts):
    # Case 1
    opts.val_loss_levels = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert(LearningStrategies.on_demand_learning(opts) == [50, 50, 50, 50, 50])
    opts.val_loss_levels = [1.0, 1.0, 2.0, 2.0, 4.0]
    assert(LearningStrategies.on_demand_learning(opts) == [25, 25, 50, 50, 100])

test_fixated_easy(opts)
test_fixated_hard(opts)
test_on_demand_learning(opts)
