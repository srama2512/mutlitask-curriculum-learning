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

    def build_queue(self,levels,num_threads=8,train=True):
        """
        levels = [0,0,0,0,1,1,1,1,2,2,2] like... tf.Variable having shape (BATCH_SIZE,)
        """
        batch_size = levels.get_shape().as_list()[0]
        target = self.trains if train else self.vals

        paths,cats = zip(*target)
        fname,l = tf.train.slice_input_producer([paths,cats],num_epochs=None,shuffle=True)
        binary = tf.read_file(fname)
        image = tf.image.decode_jpeg(binary,channels=3)
        pp_image = self._preprocess_basic(image,train)

        def _fill_tasks(elems):
            level, pp_image = elems

            def _raise():
                assert_op = tf.Assert(False,['Undefined Level'])
                with tf.control_dependencies([assert_op]):
                    return pp_image

            task_image = tf.case({
                tf.equal(level,0): lambda :self._preprocess_fill(pp_image,hole_size=(1,6),train=train),
                tf.equal(level,1): lambda :self._preprocess_fill(pp_image,hole_size=(7,12),train=train),
                tf.equal(level,2): lambda :self._preprocess_fill(pp_image,hole_size=(13,18),train=train),
                tf.equal(level,3): lambda :self._preprocess_fill(pp_image,hole_size=(19,24),train=train),
                tf.equal(level,4): lambda :self._preprocess_fill(pp_image,hole_size=(25,30),train=train)},
                default= _raise)
            return task_image

        # Build image batch
        if( train ):
            fnames, labels, pp_images = tf.train.shuffle_batch(
                [fname, l, pp_image],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=10*batch_size,
                min_after_dequeue=5*batch_size)
        else :
            fnames, labels, pp_images= tf.train.shuffle_batch(
                [fname, l, pp_image],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=10*batch_size,
                min_after_dequeue=5*batch_size)

        task_images = tf.map_fn(_fill_tasks,[levels,pp_images],dtype=tf.float32,back_prop=False)
        task_images = tf.stack(task_images,axis=0)

        pp_images = tf.transpose(pp_images,(0,3,1,2)) #NHWC -> NCHW
        task_images = tf.transpose(task_images,(0,3,1,2))
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

    def _preprocess_fill(self,image,hole_size,train=True):
        def _get_mask(image_shape,hole_size):
            mask = np.zeros(image_shape,np.float32)
            hole_size = np.random.randint(hole_size[0],hole_size[1]+1)
            hole_loc = np.random.randint(0,self.size-hole_size+1,size=2)

            mask[hole_loc[0]:(hole_loc[0]+hole_size),
                 hole_loc[1]:(hole_loc[1]+hole_size)] = 1.
            return mask

        mask = tf.py_func(_get_mask,[tf.shape(image),hole_size],tf.float32,stateful=True) #stateful True because of random
        mask = tf.reshape(mask, tf.shape(image))
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

    levels = tf.convert_to_tensor(np.array([0,1,2,3,4],np.int32))
    _,_,pp_images,task_images = sun.build_queue(levels)
    tf.summary.image('images',tf.transpose(pp_images,(0,2,3,1)),max_outputs=5)
    tf.summary.image('task_images',tf.transpose(task_images,(0,2,3,1)),max_outputs=5)
    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter('./log_temp',sess.graph)
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

