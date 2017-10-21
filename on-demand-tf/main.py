import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import EncDec
from dataset import SUN397

def main(config,
         RANDOM_SEED,
         DATA_DIR,
         LOG_DIR,
         BATCH_SIZE,
         VALID_BATCH_SIZE,
         NUM_THREADS,
         MAX_EPOCH,
         TRAIN_NUM,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         SAVE_PERIOD,
         SUMMARY_PERIOD):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    sun = SUN397(DATA_DIR,TRAIN_NUM,RANDOM_SEED)
    _,_,train_y,train_x= sun.build_queue(BATCH_SIZE,NUM_THREADS)
    _,_,valid_y,valid_x= sun.build_queue(VALID_BATCH_SIZE,NUM_THREADS,train=False)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        with tf.variable_scope('params') as params:
            pass
        net = EncDec(train_x,params,True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = EncDec(valid_x,params,False)

    with tf.variable_scope('loss'):
        with tf.control_dependencies(net.update_ops):
            l2_loss = 1./tf.cast(tf.size(train_y),tf.float32)*tf.nn.l2_loss(net.pred-train_y)
        valid_loss = 1./tf.cast(tf.size(valid_y),tf.float32)*tf.nn.l2_loss(valid_net.pred-valid_y)

        # Optimizing
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(l2_loss,global_step=global_step)


    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',l2_loss)
        tf.summary.scalar('valid_loss',valid_loss),
        summary_op = tf.summary.merge_all()

        # Expensive Summaries
        extended_summary_op = tf.summary.merge([
            tf.summary.image('train_task_images',tf.transpose(train_x,(0,2,3,1)),max_outputs=5),
            tf.summary.image('train_pred',tf.transpose(net.pred,(0,2,3,1)),max_outputs=5),
            tf.summary.image('train_answer',tf.transpose(train_y,(0,2,3,1)),max_outputs=5),

            tf.summary.image('valid_task_images',tf.transpose(valid_x,(0,2,3,1)),max_outputs=5),
            tf.summary.image('valid_pred',tf.transpose(valid_net.pred,(0,2,3,1)),max_outputs=5),
            tf.summary.image('valid_answer',tf.transpose(valid_y,(0,2,3,1)),max_outputs=5),
        ])

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for epoch in tqdm(xrange(MAX_EPOCH)):
            for step in tqdm(xrange(TRAIN_NUM)):
                it,loss,_ = sess.run([global_step,l2_loss,train_op])

                if( it % SAVE_PERIOD == 0 ):
                    net.save(sess,LOG_DIR,step=it)

                if( it % SUMMARY_PERIOD == 0 ):
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary,it)

                if( it % (SUMMARY_PERIOD*10) == 0 ):
                    summary = sess.run(extended_summary_op)
                    summary_writer.add_summary(summary,it)

                tqdm.write('[%3d/%05d] Loss: %1.3f'%(epoch,step,loss))
    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)


def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/%s'%(now),
        'DATA_DIR':'datasets/SUN397',

        'BATCH_SIZE' : 100,
        'VALID_BATCH_SIZE' : 10,
        'NUM_THREADS' : 4,

        'MAX_EPOCH' : 150,
        'TRAIN_NUM' : 100000/100, #Size corresponds to one epoch
        'LEARNING_RATE' : 0.001,
        'DECAY_VAL' : 0.1,
        'DECAY_STEPS' : 75*100000/100, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'SUMMARY_PERIOD' : 10,
        'SAVE_PERIOD' : 50000,
        'RANDOM_SEED': 0,
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix

    main(config=config,**config)
