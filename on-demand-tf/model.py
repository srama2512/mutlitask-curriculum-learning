import better_exceptions
import tensorflow as tf
import numpy as np

from commons.ops import *

class EncDec():
    # Don't forget to add control_dependencies!
    def __init__(self,x,param_scope,is_training):
        with tf.variable_scope(param_scope):
            net_spec = []
            with tf.variable_scope('encoder') as enc:
                net_spec += [
                    Conv2d('conv2d_1',3,64),
                    BatchNorm('conv2d_1',64),
                    Lrelu(),
                    Conv2d('conv2d_2',64,128),
                    BatchNorm('conv2d_2',128),
                    Lrelu(),
                    Conv2d('conv2d_3',128,256),
                    BatchNorm('conv2d_3',256),
                    Lrelu(),
                    Conv2d('conv2d_4',256,512),
                    BatchNorm('conv2d_4',512),
                    Lrelu(),
                ]
            with tf.variable_scope('channel-wise'):
                net_spec += [
                    DepthConv2d('fc',512,16,4,4,1,1,padding='VALID'),
                    lambda t,**kwargs : tf.reshape(t,[-1,512,4,4])
                ]
            with tf.variable_scope('decoder'):
                net_spec += [
                    TransposedConv2d('trans_conv2d_1',512,256),
                    BatchNorm('trans_conv2d_1',256),
                    lambda t,**kwargs : tf.nn.relu(t),
                    TransposedConv2d('trans_conv2d_2',256,128),
                    BatchNorm('trans_conv2d_2',128),
                    lambda t,**kwargs : tf.nn.relu(t),
                    TransposedConv2d('trans_conv2d_3',128,64),
                    BatchNorm('trans_conv2d_3',64),
                    lambda t,**kwargs : tf.nn.relu(t),
                    TransposedConv2d('trans_conv2d_4',64,3),
                    BatchNorm('trans_conv2d_4',3),
                    lambda t,**kwargs : tf.tanh(t),
                ]

        kwargs = {'is_training':is_training} #BatchNorm Params
        with tf.variable_scope('net') as net_scope:
            _t = x
            for block in net_spec :
                _t = block(_t,**kwargs)

        if( is_training ) :
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,net_scope.name)
            self.pred = _t
            #for op in self.update_ops:
            #    print op
            #with tf.control_dependencies(self.update_ops):
            #    self.pred = tf.identity(_t)
        else:
            self.pred = _t

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        #for name,var in save_vars.items():
        #    print(name,var)
        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)

    x = tf.placeholder(tf.float32,[None,3,64,64])
    with tf.variable_scope('c1'):
        with tf.variable_scope('params') as params:
            pass
        c1 = EncDec(x,params,True)

    with tf.variable_scope('c2'):
        params.reuse_variables()
        c2 = EncDec(x,params,False)

    #print (c1.pred)

    #for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) :
    #    print var

    #betas = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,params.name) if 'beta' in var.name]
    #gammas = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,params.name) if 'gamma' in var.name]
    #mms = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,params.name) if 'moving_mean' in var.name]
    #mvars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,params.name) if 'moving_variance' in var.name]
    #print(betas)
    #print(gammas)
    #print(mms)
    #print(mvars)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess = tf.Session()
    sess.graph.finalize()
    sess.run(init_op)

    inp = np.random.random((10,3,64,64)).astype(np.float32)

    t1 = sess.run( c1.pred, feed_dict={x:inp})

    for _ in range(1000):
        t2 = sess.run( c1.pred, feed_dict={x:inp})
        print( np.linalg.norm(t1-t2) )
    c1.save(sess,'./test')

    t1 = sess.run( c2.pred, feed_dict={x:inp})
    print( np.linalg.norm(t1-t2) )
    np.save(open('t1.npy','w'),t2)

    #inp = np.random.random((10,3,64,64)).astype(np.float32)

    #x = tf.placeholder(tf.float32,[None,3,64,64])
    #with tf.variable_scope('train'):
    #    with tf.variable_scope('params') as params:
    #        pass
    #    test = EncDec(x,params,False)
    #init_op = tf.group(tf.global_variables_initializer(),
    #                tf.local_variables_initializer())
    #sess = tf.Session()
    #sess.graph.finalize()
    #sess.run(init_op)
    #test.load(sess,'./test/last.ckpt')

    #t2 = np.load(open('t1.npy'))
    #t1 = sess.run( test.pred, feed_dict={x:inp})
    #print( np.linalg.norm(t1-t2) )
