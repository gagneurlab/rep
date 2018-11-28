"""
Global variables
"""
import os


valid_chr = ['chr2', 'chr3', 'chr4']
test_chr = ['chr1', 'chr8', 'chr9']


def get_data_dir():
    """Returns the data directory
    """
    import inspect
    import os
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_path = os.path.dirname(os.path.abspath(filename))
    DATA = os.path.join(this_path, "../data")
    if not os.path.exists(DATA):
        raise ValueError(DATA + " folder doesn't exist")
    return os.path.abspath(DATA)


def create_tf_session(visiblegpus, per_process_gpu_memory_fraction=0.45):
    import os
    import tensorflow as tf
    import keras.backend as K
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    # session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    session = tf.Session(config=session_config)
    K.set_session(session)
    return session


# def get_exp_dir(exp):
#     """Get the experiment directory
#     """
#     ddir = get_data_dir()
#     url = "http://mitra.stanford.edu/kundaje/avsec/chipnexus/"
#     www_path = f"exp/chipnexus/{exp}"
#     return f"{ddir}/processed/chipnexus/exp/{exp}", \
#         f"{ddir}/www/{www_path}", \
#         f"{url}/{www_path}"
