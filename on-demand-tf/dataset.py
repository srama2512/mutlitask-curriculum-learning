import better_exceptions
import numpy as np
import tensorflow as tf
import random
import os

class SUN397:
    def __init__(self,data_dir,train_num=100000,seed=0,
                 size=64,):
        classNames = [ l.strip()[1:] for l in open(os.path.join(data_dir,'ClassName.txt')).readlines() ]
        self.files = [(os.path.join(data_dir,c,f),i)
                 for i,c in enumerate(classNames)
                    for f in os.listdir(os.path.join(data_dir,c))
                        if f.endswith(".jpg")]
        old_state = random.getstate()
        random.seed(seed)
        random.shuffle(self.files)
        random.setstate(old_state)

        #print len(files), files[0]
        self.trains = self.files[:train_num]
        self.vals = self.files[train_num:]

        self.size = size

    def build_queue(self,batch_size,num_threads=8,train=True):
        target = self.trains if train else self.vals

        paths,cats = zip(*target)
        fname,l = tf.train.slice_input_producer([paths,cats],num_epochs=None,shuffle=True)
        binary = tf.read_file(fname)
        image = tf.image.decode_jpeg(binary,channels=3)
        pp_image = self._preprocess_basic(image,train)
        task_image = self._preprocess_fill(pp_image,hole_size=20,train=train)

        pp_image = tf.transpose(pp_image,(2,0,1))
        task_image = tf.transpose(task_image,(2,0,1))

        # Build image batch
        if( train ):
            fnames, labels, pp_images, task_images = tf.train.shuffle_batch(
                [fname, l, pp_image, task_image],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=10*batch_size,
                min_after_dequeue=5*batch_size)
        else :
            fnames, labels, pp_images, task_images = tf.train.batch(
                [fname, l, pp_image, task_image],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=10*batch_size,
                allow_smaller_final_batch=True,)
        #tf.summary.image('images',tf.transpose(pp_images,(0,2,3,1)),max_outputs=5)
        #tf.summary.image('task_images',tf.transpose(task_images,(0,2,3,1)),max_outputs=5)

        return fnames,labels,pp_images,task_images

    def _preprocess_basic(self,image,train=True):
        if(train):
            _t = tf.image.random_flip_left_right(image)
            bbox_begin, bbox_size, _ = \
                tf.image.sample_distorted_bounding_box(tf.shape(_t),
                                                       bounding_boxes=[[[0.,0.,1.,1.]]],
                                                       min_object_covered=0.75,
                                                       aspect_ratio_range=(0.75,1.33),
                                                       area_range=(0.5,1.0),
                                                       use_image_if_no_bounding_boxes=True)
            cropped_image = tf.slice(_t, bbox_begin, bbox_size)
        else:
            cropped_image = tf.image.central_crop(image, central_fraction=0.85)

        _t = tf.image.resize_images(cropped_image, [self.size, self.size], tf.image.ResizeMethod.BICUBIC)
        _t = tf.cast(_t,tf.float32) / 255.0
        _t = tf.subtract(_t, 0.5)
        _t = tf.multiply(_t, 2.0)
        return tf.reshape(_t,[self.size,self.size,3])
    def _preprocess_fill(self,image,hole_size=20,train=True):
        mask  = tf.Variable(tf.zeros_like(image),dtype=tf.float32)
        masking = mask[(self.size//2-hole_size//2):(self.size//2+hole_size//2),
                       (self.size//2-hole_size//2):(self.size//2+hole_size//2)].assign(np.ones((hole_size,hole_size,3),np.float32))
        with tf.control_dependencies([masking]):
            task_image = (1.-mask)*image + mask*(tf.ones_like(image)*-1)
            return task_image
    def _preprocess_noisy(self,train=True):
        pass
    def _preprocess_blur(self,train=True):
        pass
    def _preprocess_interpol(self,train=True):
        pass


if __name__ == "__main__":
    from tqdm import tqdm
    sun = SUN397('datasets/SUN397')

    _,_,pp_images,task_images = sun.build_queue(16)
    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter('./log',sess.graph)
    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(100)):
            x,y,summary_str = sess.run([pp_images,task_images,summary_op])
            summary_writer.add_summary(summary_str,step)
    except Exception as e:
        coord.request_stop(e)
    finally :

        coord.request_stop()
        coord.join(threads)
    print x.shape, y.shape

