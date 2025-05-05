# QTune-Torch

PyTorch implementation of QTune for automatic database configuration tuning.

## Features

[✓] Automatic parameter tuning based on workload + state features (environment.py)

[✓] Solved hardcoding issues (database, benchmark, knob categories/ranges/types, system state metrics, prediction model, model size, etc.)

[✓] Provides pre-trained data and models (torch_model.py)

[✓] OLAP/OLTP test sets (JOB, Sysbench, TPC-H)

[✓] Training curve visualization (draw.py), including random exploration and model prediction performance (latency/throughput)

[✓] Zero-Order optimization support for actor network training

[todo] Support restart/non-restart parameters simultaneously

[todo] Support template-based online parameter tuning; end-to-end training of tuning models and workload encoding models

[todo] Integrate existing rule-based pre-tuning


## Setup Steps

1. Install MySQL 5.7 database

   ```bash
   docker run --name mysql-qt -e MYSQL_ROOT_PASSWORD=123456 \
   -v $(pwd)/docker_conf_sql5.7/init.sql:/docker-entrypoint-initdb.d/init.sql \
   -v $(pwd)/docker_conf_sql5.7/sql.cnf:/etc/mysql/conf.d/sql.cnf \
   -p 3307:3306 -d mysql:5.7
   ```

   * For MySQL root user remote access permissions, refer to: [MySQL root remote access permissions](https://www.cnblogs.com/goxcheer/p/8797377.html)

   * Add a new line `sql_mode=NO_ENGINE_SUBSTITUTION` to `my.cnf` or `my.ini`, and restart mysqld

2. Upload standard test set code to the database server and install:
   * Sysbench: [Installation guide](https://blog.csdn.net/cxin917/article/details/81557453) (Install on database server)
     ```bash
     sysbench oltp_read_only prepare --db-driver=mysql --tables=20 --table-size=8000000 --mysql-host=127.0.0.1 --mysql-user=root --mysql-password=123456 --mysql-port=3307 --mysql-db='sysbench' --mysql-storage-engine=innodb --mysql-db='sysbench'  --time=150  --range-size=10 --rand-type=uniform --report-interval=10
     ```
   * JOB: [Installation guide](https://blog.csdn.net/cxin917/article/details/81557453) (IMDB dataset can be provided via cloud if download is slow)
   * Enable mysqllog to obtain workload.
   * Disable mysqllog when finished.

3. Place the tuning code on a server that can connect to the database, configure Python 3.6+ dependencies. Install required packages:
   ```bash
   pip3 install -r requirements.txt
   ```

4. Configure parameters:
   Copy config_example.ini to create a new file named config.ini
   ```bash
   cp config_example.ini config.ini
   ```
   Modify parameters in config.ini
   workload_file_path will be obtained from the specified file, the content can refer to workload_file_example.txt file.

5. Run the code:
   ```bash
   python3 main.py
   ```

6. Manually draw training.png to observe training effects:
   ```bash
   python3 draw.py res_predict-1623852012,res_random-1623852012 latency
   ```

## Zero-Order Optimization

This implementation supports zero-order optimization for training the actor network. Configure the optimization parameters in the `config.ini` file:

```ini
[zero_order]
noise_std = 0.1
noise_decay = 0.99
lr_decay = 0.99
decay_step = 50
norm_rewards = true
```

## Troubleshooting

**Q:** "mysql cannot connect from remote host"

**A:** https://devanswers.co/cant-connect-mysql-server-remotely/

**Q:** JOB queries take very long time to run

**A:** Delete all foreign keys and create indexes (modify fkindexes.sql); change scripts to parallel execution; select partial SQL execution

**Q:** "ImportError: attempted relative import with no known parent package"

**A:** https://stackoverflow.com/questions/16981921/relative-imports-in-python-3

**Q:** pd.read_csv: "ValueError: cannot convert float NaN to integer"

**A:** 1. Use sep = "\t"; 2. Check encoding style (utf-8)



