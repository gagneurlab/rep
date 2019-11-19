def rechunk_iter(data_iter, subbatch_size, drop_last=False):
    shift = 0
    last_batch = None
    for batch in data_iter:
        if shift != 0:
            yield np.concatenate([last_batch, batch[:shift]], axis=0)

        batch_size = np.shape(batch)[0] - shift # substract already yielded part of batch
        num_subbatches = batch_size // subbatch_size # number of full batches
        for i in range(num_subbatches):
            subbatch_start = shift + i * subbatch_size # batch starts at subbatch number * subbatch size, ignoring the yielded start batch
            yield batch[subbatch_start: subbatch_start + subbatch_size]

        remain = batch_size - num_subbatches * subbatch_size # + shift - shift, cancels out
        if remain != 0:
            last_batch = batch[shift + num_subbatches * subbatch_size:]
            shift = subbatch_size - remain

    if not drop_last and shift != 0:
        yield last_batch

import threading
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

# @threadsafe_generator

def dask_batch_iter(data, drop_last=False, batch_size=None):
    if batch_size == None:
        batch_size = data.chunksize[0]
    num_samples = np.shape(data)[0]
    num_batches = num_samples // batch_size

    for i in range(num_batches):
        batch_start = i * batch_size
        yield dask.compute(data[batch_start: batch_start + batch_size])[0]

    if not drop_last and num_samples % batch_size != 0:
        yield dask.compute(data[num_batches * batch_size:])[0]


def generator_fn(training_data, dim="observations", batch_size=256):
    print(batch_size)
    features = training_data.c_features.transpose("observations", "features", transpose_coords=False).data
    target = training_data.normppf.transpose("observations", "subtissue", transpose_coords=False).data

    batch_size = features.chunksize[0]
    for i in range(training_data.dims[dim] // batch_size):
        # batch = training_data.isel(observations=slice(i, i + batch_size))
        yield (
            features[start: start + batch_size],
            target[start: start + batch_size],
        )

