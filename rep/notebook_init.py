import os
import sys

import psutil

import json

import joblib
import dask

import sys


__all__ = (
    "notebook_logger",
    "memory_limit",
    "MEMORY_LIMIT",
    "set_cpu_count_env",
    "init_ray",
    "init_dask",
)


def notebook_logger(name, log_file):
    import logging

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename=log_file, mode='a')
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)

    log.addHandler(file_handler)
    log.addHandler(stdout_handler)
    log.propagate = False

    return log


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


    if adjust_env:
        # ensure initialization of workers' env variables with one core
        set_cpu_count_env(n_cpu=1)

    # Start Ray.
    # Tip: If you're connecting to an existing cluster, use ray.init(address="auto").
    ray.init(
        _memory=MEMORY_LIMIT * 0.7,
        object_store_memory=MEMORY_LIMIT * 0.3,
        num_cpus=n_cpu,
        _temp_dir=os.environ["TMPDIR"],
        _system_config={
            "automatic_object_spilling_enabled": True,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": spill_dir}},
            )
        },
    )
    if adjust_env:
        # reset env variables with one core
        set_cpu_count_env(n_cpu=n_cpu)

    from ray.util.dask import ray_dask_get, dataframe_optimize
    from ray.util.joblib import register_ray
    register_ray()

    def dask_init_ray():
        dask.config.set(
            scheduler=ray_dask_get,
            dataframe_optimize=dataframe_optimize,
            shuffle='tasks',
            # no idea how to set max_branch globally
            # max_branch=float("inf"),
        )
    dask_init_ray()
    ray.worker.global_worker.run_function_on_all_workers(lambda args: dask_init_ray())

    if adjust_env:
        def adjust_worker_env_fn(args):
            if "worker" in args:
                worker = args["worker"]
            else:
                worker = ray.worker.global_worker

            if worker.mode in (ray.LOCAL_MODE, ray.SCRIPT_MODE):
                # worker is driver; do not change environment
                return
            else:
                # make sure that OMP_NUM_THREADS, etc. is set to 1 on all workers but not on the driver
                set_cpu_count_env(n_cpu=1)

        ray.worker.global_worker.run_function_on_all_workers(adjust_worker_env_fn)

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


def init_spark(
        app_name="REP",
        memory=MEMORY_LIMIT,
        memory_factor=0.9,
        n_cpu=joblib.cpu_count(),
        max_failures=4,
        num_shuffle_partitions=None,
        tmpdir=None,
        max_result_size=None,
        additional_packages=(),
        additional_jars=(),
        enable_glow=True,
        enable_delta=False,
        enable_delta_cache=True,
        # enable_psql=True,
        enable_sqlite=True,
    ):
    # parse memory
    if isinstance(memory, str):
        import humanfriendly
        memory = humanfriendly.parse_size(memory)

    # reduce total amount of memory that the Spark driver is allowed to use
    memory = memory * memory_factor

    from pyspark.sql import SparkSession

    os.environ['PYSPARK_SUBMIT_ARGS'] = " ".join([
        f'--driver-memory {int(memory // 1024)}k',
        'pyspark-shell'
    ])

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.local.dir", os.environ.get("TMP") if tmpdir is None else tmpdir)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.cbo.enabled", "true")
        .config("spark.sql.cbo.joinReorder.enabled", "true")
        .config("spark.driver.maxResultSize", f"{int(memory)}b" if max_result_size is None else max_result_size)
        .config('spark.sql.caseSensitive', "true")
    )

    if max_failures is not None:
        spark = spark.config("spark.task.maxFailures", max_failures)
    if num_shuffle_partitions is not None:
        spark = spark.config("spark.sql.shuffle.partitions", num_shuffle_partitions)

    packages=[*additional_packages]
    jars=[*additional_jars]
    if enable_glow:
        packages.append("io.projectglow:glow-spark3_2.12:1.1.0")
        spark = spark.config("spark.hadoop.io.compression.codecs", "io.projectglow.sql.util.BGZFCodec")
    if enable_delta:
        packages.append("io.delta:delta-core_2.12:1.0.0")
        spark = (
            spark
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        if enable_delta_cache:
            # CAUTION: only enable when local storage is actually on local SSD!!
            spark = spark.config("spark.databricks.io.cache.enabled", "true")
    if enable_glow:
        packages.append("io.projectglow:glow-spark3_2.12:1.1.0")
    # if enable_psql:
    #     packages.append("org.postgresql:postgresql:42.2.12")
    if enable_sqlite:
        packages.append("org.xerial:sqlite-jdbc:3.36.0.1")

    if len(packages) > 0:
        spark = spark.config("spark.jars.packages", ",".join(packages))
    if len(jars) > 0:
        spark = spark.config("spark.jars", ",".join(jars))

    # spawn the session
    spark = spark.getOrCreate()

    if enable_glow:
        import glow
        glow.register(spark)

    return spark


def setup_plot_style():
    import plotnine as pn
    import matplotlib
    import seaborn as sns

    pn.themes.theme_set(pn.theme_bw)

    matplotlib.style.use('seaborn')
    matplotlib.rcParams['figure.dpi'] = 300
    matplotlib.rcParams['figure.figsize'] = [12, 8]
    matplotlib.rcParams["savefig.dpi"] = 450


