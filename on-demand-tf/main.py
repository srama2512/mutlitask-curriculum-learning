import better_exceptions
import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm

from model import EncDec
from dataset import SUN397

def main(config,
         RANDOM_SEED,
         DATA_DIR,
         LOG_DIR,
         TASK_NUM,
         BATCH_SIZE,
         DEMAND_TEST_ITER,
         NUM_THREADS,
         MAX_EPOCH,
         TRAIN_NUM,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         SAVE_PERIOD,
         SUMMARY_PERIOD):
    assert(TASK_NUM in [0,1,2])
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    sun = SUN397(DATA_DIR,TRAIN_NUM,RANDOM_SEED)

    train_batch_levels = tf.placeholder(tf.int32,[BATCH_SIZE,])
    _,_,train_y,train_x= sun.build_queue(tf.convert_to_tensor(np.array([TASK_NUM]*BATCH_SIZE)),train_batch_levels,NUM_THREADS)
    _,_,valid_y,valid_x= sun.build_queue(tf.convert_to_tensor(np.array([TASK_NUM]*10)),tf.convert_to_tensor(np.array([0,1,2,3,4]*2)),NUM_THREADS,train=False)
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
        # mse & psnr
        def _imagify(ims,cast=tf.uint8):
            ims = tf.transpose(ims,(0,2,3,1))
            return tf.cast((ims / 2 + 0.5)*255.0,cast)
        def _psnr(mse):
            return (20. * tf.log(255.) - 10. * tf.log(mse)) / tf.log(10.)


        mse_op = tf.reduce_mean((_imagify(net.pred,tf.float32) - _imagify(train_y,tf.float32))**2,axis=[1,2,3])
        psnr_op = _psnr(mse_op)
        valid_mse_op = tf.reduce_mean((_imagify(valid_net.pred,tf.float32) - _imagify(valid_y,tf.float32))**2,axis=[1,2,3])
        valid_psnr_op = _psnr(valid_mse_op)
        tf.summary.scalar('mse',tf.reduce_mean(mse_op))
        tf.summary.scalar('psnr',tf.reduce_mean(psnr_op))
        tf.summary.scalar('valid_mse',tf.reduce_mean(valid_mse_op))
        tf.summary.scalar('valid_psnr',tf.reduce_mean(valid_psnr_op))

        summary_op = tf.summary.merge_all()


        # Expensive Summaries
        extended_summary_op = tf.summary.merge([
            tf.summary.image('train_task_images',_imagify(train_x),max_outputs=5),
            tf.summary.image('train_pred',_imagify(net.pred),max_outputs=5),
            tf.summary.image('train_answer',_imagify(train_y),max_outputs=5),

            tf.summary.image('valid_task_images',_imagify(valid_x),max_outputs=5),
            tf.summary.image('valid_pred',_imagify(valid_net.pred),max_outputs=5),
            tf.summary.image('valid_answer',_imagify(valid_y),max_outputs=5),
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

    ratios = [0.2,0.2,0.2,0.2,0.2]
    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for epoch in tqdm(xrange(MAX_EPOCH)):
            levels = [ l for l,ratio in enumerate(ratios)
                        for _ in xrange(int(ratio*BATCH_SIZE))]
            while(len(levels)<BATCH_SIZE):
                levels.append(4)
            random.shuffle(levels) # For randomized summary....

            tqdm.write('[%3d] Current Ratio %s'%(epoch,str(ratios)))
            for step in tqdm(xrange(TRAIN_NUM)):
                it,loss,mse,psnr,_ = sess.run([global_step,l2_loss,mse_op,psnr_op,train_op],feed_dict={train_batch_levels:levels})

                if( it % SAVE_PERIOD == 0 ):
                    net.save(sess,LOG_DIR,step=it)

                if( it % SUMMARY_PERIOD == 0 ):
                    summary = sess.run(summary_op,feed_dict={train_batch_levels:levels})
                    summary_writer.add_summary(summary,it)

                if( it % (SUMMARY_PERIOD*10) == 0 ):
                    summary = sess.run(extended_summary_op,feed_dict={train_batch_levels:levels})
                    summary_writer.add_summary(summary,it)

                tqdm.write('[%3d/%5d] Loss: %1.3f MSE: %1.3f PSNR: %1.3f'%(epoch,step,loss,np.mean(mse),np.mean(psnr)))

            # calculate next batch ratios
            valid_psnr = []
            for _ in tqdm(xrange(DEMAND_TEST_ITER)):
                valid_psnr.append(sess.run(valid_psnr_op).reshape(2,5))
            valid_psnr= np.mean(np.concatenate(valid_psnr,axis=0),axis=0)
            ratios = (1/valid_psnr) / np.sum(1/valid_psnr)
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
        'LOG_DIR':'./log_temp/%s'%(now),
        'DATA_DIR':'datasets/SUN397',

        'TASK_NUM' : 1,
        'BATCH_SIZE' : 100,
        'DEMAND_TEST_ITER' : 50, # 500 samples to decide next difficulty ratio
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
