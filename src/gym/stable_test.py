import tensorflow as tf
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('pcc_ori_model_3.ckpt.meta') #載入網路結構
    new_saver.restore(sess, tf.train.latest_checkpoint('./')) #載入最近一次儲存的ckpt
    print(new_saver)
    sess.run(tf.global_variables_initializer())
    print(sess.run('obs:0'))