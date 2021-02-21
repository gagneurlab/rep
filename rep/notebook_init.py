import os
import sys

import psutil

import json

import joblib
import dask

import sys


__all__ = (
    "memory_limit",
    "MEMORY_LIMIT",
    "set_cpu_count_env",
    "init_ray",
    "init_dask",
)


def memory_limit():
    """Get the memory limit (in bytes) for this system.

    Takes the minimum value from the following locations:

    - Total system host memory
    - Cgroups limit (if set)
    - RSS rlimit (if set)
    """
    limit = psutil.virtual_memory().total

    # Check cgroups if available
    if sys.platform == "linux":
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                cgroups_limit = int(f.read())
            if cgroups_limit > 0:
                limit = min(limit, cgroups_limit)
        except Exception:
            pass

    # Check rlimit if available
    try:
        import resource

        hard_limit = resource.getrlimit(resource.RLIMIT_RSS)[1]
        if hard_limit > 0:
            limit = min(limit, hard_limit)
    except (ImportError, OSError):
        pass

    return limit


MEMORY_LIMIT = memory_limit()


def set_cpu_count_env(n_cpu=joblib.cpu_count()):
    for var in [
        "GOTO_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
    ]:
        os.environ[var] = str(n_cpu)


def init_ray(adjust_env=True, n_cpu=joblib.cpu_count()):
    import ray

    spill_dir = os.path.join(os.environ["TMPDIR"], "ray_spill")
    try:
        os.mkdir(spill_dir)
    except:
        pass

    # Start Ray.
    # Tip: If you're connecting to an existing cluster, use ray.init(address="auto").
    ray.init(
        _memory=MEMORY_LIMIT * 0.7,
        object_store_memory=MEMORY_LIMIT * 0.3,
        num_cpus=joblib.cpu_count(),
        _temp_dir=os.environ["TMPDIR"],
        _system_config={
            "automatic_object_spilling_enabled": True,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": spill_dir}},
            )
        },
    )

    from ray.util.dask import ray_dask_get
    from ray.util.joblib import register_ray
    register_ray()

    dask.config.set(scheduler=ray_dask_get)

    if adjust_env:
        # make sure that OMP_NUM_THREADS, etc. is set to 1 on all workers
        ray.worker.global_worker.run_function_on_all_workers(lambda x: set_cpu_count_env(n_cpu=1))
        # set number of threads in main process' env variables
        set_cpu_count_env(n_cpu)


    return ray.cluster_resources()


def init_dask(adjust_env=True, lifetime_restart=False):
    if adjust_env:
        set_cpu_count_env()

    import dask.distributed

    # from dask.cache import Cache
    # cache = Cache(8e9)  # Leverage eight gigabytes of memory
    # cache.register()

    # dask.config.set({'temporary_directory': os.environ['TMP']})
    # dask.config.set({"distributed.scheduler.allowed-failures": 25})

    cluster = dask.distributed.LocalCluster(
        threads_per_worker=8,
        env={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_MAX_THREADS": "1",
        }
    )
    if lifetime_restart:
        client = dask.distributed.Client(cluster, lifetime="20 minutes", lifetime_restart=True)
    else:
        client = dask.distributed.Client(cluster)

    return client
