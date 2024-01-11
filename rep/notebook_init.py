import os
import sys

import json
import psutil
import joblib

__all__ = (
    "notebook_logger",
    "memory_limit",
    "MEMORY_LIMIT",
    "set_cpu_count_env",
    "init_ray",
    "init_dask",
    "init_spark",
    "init_spark_on_ray",
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


def init_ray(
    adjust_env=True,
    n_cpu=joblib.cpu_count(),
    cluster_addr=None,
    plasma_store_memory_fraction=0.05
):
    import ray
    import dask
    from ray.util.dask import ray_dask_get, dataframe_optimize
    from ray.util.joblib import register_ray

    if cluster_addr is None:
        cluster_addr = os.environ.get("RAY_ADDRESS")

    spill_dir = os.path.join(os.environ["TMPDIR"], "ray_spill")
    try:
        os.mkdir(spill_dir)
    except:
        pass

    worker_process_setup_hooks = []
    
    def dask_init_ray():
        dask.config.set({
            "scheduler": ray_dask_get,
            "dataframe_optimize": dataframe_optimize,
            # no idea how to set max_branch globally
            # max_branch=float("inf"),
        })
    worker_process_setup_hooks.append(dask_init_ray)

    def adjust_worker_env():
        # make sure that OMP_NUM_THREADS, etc. is set to 1 on all workers but not on the driver
        set_cpu_count_env(n_cpu=1)
    if adjust_env:
        worker_process_setup_hooks.append(adjust_worker_env)

    if cluster_addr is not None:
        # we're in client mode
        ray_context = ray.init(
            address=cluster_addr,
        )
    else:
        def worker_setup_fn():
            for fn in worker_process_setup_hooks:
                fn()

        if adjust_env:
            # ensure initialization of workers' env variables with one core
            set_cpu_count_env(n_cpu=1)

        # Start Ray.
        ray_context = ray.init(
            _memory=MEMORY_LIMIT * (1 - plasma_store_memory_fraction),
            object_store_memory=MEMORY_LIMIT * plasma_store_memory_fraction,
            num_cpus=n_cpu,
            _temp_dir=os.environ["TMPDIR"],
            _system_config={
                "automatic_object_spilling_enabled": True,
                "object_spilling_config": json.dumps(
                    {"type": "filesystem", "params": {"directory_path": spill_dir}},
                )
            },
            runtime_env={
                "worker_process_setup_hook": worker_setup_fn
            }
        )
        if adjust_env:
            # reset env variables with one core
            set_cpu_count_env(n_cpu=n_cpu)

    register_ray()
    dask_init_ray()

    return ray_context


def init_dask(adjust_env=True, lifetime_restart=False):
    if adjust_env:
        set_cpu_count_env()

    import dask
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


def _spark_conf(
        max_failures=4,
        num_shuffle_partitions=None,
        tmpdir=None,
        max_result_size=None,
        additional_packages=(),
        additional_jars=(),
        additional_extensions=(),
        enable_glow=False,
        enable_iceberg=True,
        enable_delta=False,
        enable_delta_cache=True,
        # enable_psql=True,
        enable_sqlite=True,
        enable_kryo_serialization=True,
        kryo_buffer_max="512m"
):
    from importlib.metadata import version
    pyspark_version = version('pyspark')

    config = {}

    config["spark.local.dir"] = os.environ.get("TMP") if tmpdir is None else tmpdir
    config["spark.sql.execution.arrow.pyspark.enabled"] = "true"
    config["spark.sql.adaptive.enabled"] = "true"
    config["spark.sql.cbo.enabled"] = "true"
    config["spark.sql.cbo.joinReorder.enabled"] = "true"
    if max_result_size is not None:
        config["spark.driver.maxResultSize"] = max_result_size
    config['spark.sql.caseSensitive'] = "true"

    if max_failures is not None:
        config["spark.task.maxFailures"] = max_failures
    if num_shuffle_partitions is not None:
        config["spark.sql.shuffle.partitions"] = num_shuffle_partitions
    if enable_kryo_serialization:
        config["spark.serializer"] = "org.apache.spark.serializer.KryoSerializer"
        config["spark.kryo.unsafe"] = "true"
        config["spark.kryoserializer.buffer.max"] = kryo_buffer_max

    packages = [*additional_packages]
    jars = [*additional_jars]
    extensions = [*additional_extensions]
    if enable_glow:
        if pyspark_version.startswith("3.1."):
            packages.append("io.projectglow:glow-spark3_2.12:1.1.2")
        elif pyspark_version.startswith("3.2."):
            packages.append("io.projectglow:glow-spark3_2.12:1.2.1")
        else:
            raise ValueError(f"Unknown glow version for PySpark v{pyspark_version}!")
        config["spark.hadoop.io.compression.codecs"] = "io.projectglow.sql.util.BGZFCodec"
    if enable_delta:
        if pyspark_version.startswith("3.1."):
            packages.append("io.delta:delta-core_2.12:1.0.1")
        elif pyspark_version.startswith("3.2."):
            packages.append("io.delta:delta-core_2.12:2.0.1")
        elif pyspark_version.startswith("3.3."):
            packages.append("io.delta:delta-core_2.12:2.3.0")
        elif pyspark_version.startswith("3.4."):
            packages.append("io.delta:delta-core_2.12:2.4.0")
        else:
            raise ValueError(f"Unknown glow version for PySpark v{pyspark_version}!")

        extensions.append("io.delta.sql.DeltaSparkSessionExtension")
        config["spark.sql.catalog.spark_catalog"] = "org.apache.spark.sql.delta.catalog.DeltaCatalog"

        if enable_delta_cache:
            # CAUTION: only enable when local storage is actually on local SSD!!
            config["spark.databricks.io.cache.enabled"] = "true"
    if enable_iceberg:
        if pyspark_version.startswith("3.1."):
            packages.append("org.apache.iceberg:iceberg-spark-runtime-3.1_2.12:1.3.0")
        elif pyspark_version.startswith("3.2."):
            packages.append("org.apache.iceberg:iceberg-spark-runtime-3.2_2.12:1.3.0")
        elif pyspark_version.startswith("3.3."):
            packages.append("org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.3.0")
        elif pyspark_version.startswith("3.4."):
            packages.append("org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.3.0")
        else:
            raise ValueError(f"Unknown glow version for PySpark v{pyspark_version}!")

        extensions.append("org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        config["spark.sql.catalog.spark_catalog"] = "org.apache.iceberg.spark.SparkSessionCatalog"
        config["spark.sql.catalog.spark_catalog.type"] = "hive"
        # config["spark.sql.catalog.local"] = "org.apache.iceberg.spark.SparkCatalog"
        # config["spark.sql.catalog.local.type"] = "hadoop"
        # config["spark.sql.catalog.local.warehouse"] = os.getcwd() + "/warehouse"

    # if enable_psql:
    #     packages.append("org.postgresql:postgresql:42.2.12")
    if enable_sqlite:
        packages.append("org.xerial:sqlite-jdbc:3.36.0.1")

    if len(packages) > 0:
        config["spark.jars.packages"] = ",".join(packages)
    if len(jars) > 0:
        config["spark.jars"] = ",".join(jars)
    if len(extensions) > 0:
        config["spark.sql.extensions"] = ",".join(extensions)


    return config


def init_spark_on_ray(
        executor_cores=16,
        executor_memory_overhead=0.95,
        total_num_cores=None,
        driver_memory=None,
        configs=None,
        spark_conf_args=None,
        **kwargs
):
    """

    :param driver_memory: driver memory in kilobyte
    :param total_num_cores: total number of cores to use;
        Will use all available cores by default.
    """
    import ray
    import raydp

    if configs is None:
        configs = {}
    if spark_conf_args is None:
        spark_conf_args = {}

    if driver_memory is None:
        driver_memory = int(MEMORY_LIMIT / 2)
    
    if total_num_cores is None:
        total_num_cores = int(ray.available_resources()["CPU"])

    # set driver memory
    os.environ['PYSPARK_SUBMIT_ARGS'] = " ".join([
        f'--driver-memory {int(driver_memory // 1024)}k',
        'pyspark-shell'
    ])

    spark_conf_args["enable_glow"] = spark_conf_args.get("enable_glow", False)
    spark_conf_args["max_result_size"] = spark_conf_args.get("max_result_size", driver_memory)
    spark_conf_args["num_shuffle_partitions"] = spark_conf_args.get("num_shuffle_partitions", total_num_cores * 2)

    spark_conf = _spark_conf(**spark_conf_args)
    configs = {
        "spark.default.parallelism": total_num_cores,
        **spark_conf,
        **configs,
    }

    spark = raydp.init_spark(
        app_name="raydp",
        num_executors=int(ray.available_resources()["CPU"] / executor_cores),
        executor_cores=executor_cores,
        executor_memory=int(
            (ray.available_resources()["memory"] / total_num_cores) * executor_cores * executor_memory_overhead
        ),
        configs=configs,
        #    configs={"raydp.executor.extraClassPath": os.environ["SPARK_HOME"] + "/jars/*"},
        **kwargs,
    )
    if spark_conf_args["enable_glow"]:
        import glow
        glow.register(spark)

    # spark.stop = raydp.stop_spark

    return spark


def init_spark(
        app_name="REP",
        memory=MEMORY_LIMIT,
        memory_factor=0.9,
        enable_glow=False,
        **kwargs,
):
    # parse memory
    if isinstance(memory, str):
        import humanfriendly
        memory = humanfriendly.parse_size(memory)

    # reduce total amount of memory that the Spark driver is allowed to use
    memory = memory * memory_factor

    kwargs["max_result_size"] = kwargs.get("max_result_size", f"{int(memory)}b")

    from pyspark.sql import SparkSession

    os.environ['PYSPARK_SUBMIT_ARGS'] = " ".join([
        f'--driver-memory {int(memory // 1024)}k',
        'pyspark-shell'
    ])

    spark = (
        SparkSession.builder
        .appName(app_name)
    )

    config = _spark_conf(
        enable_glow=enable_glow,
        **kwargs
    )
    for k, v in config.items():
        spark = spark.config(k, v)

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

    matplotlib.style.use('seaborn-v0_8')
    matplotlib.rcParams['figure.dpi'] = 300
    matplotlib.rcParams['figure.figsize'] = [12, 8]
    matplotlib.rcParams["savefig.dpi"] = 450
