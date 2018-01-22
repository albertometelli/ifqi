import baselines.common.tf_util as tf_util
import tensorflow as tf

def get_session():
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    return sess
