import threading
import time
import queue
import pymysql
from concurrent.futures import ThreadPoolExecutor
from configs import parse_args
from dbutils.pooled_db import PooledDB
from tqdm import tqdm

args = parse_args()
lock = threading.Lock()
total_lat = 0
error_query_num = 0

POOL = None


# 把任务放入队列中
class Producer(threading.Thread):
    def __init__(self, name, queue, workload):
        self.__name = name
        self.__queue = queue
        self.workload = workload
        super(Producer, self).__init__()

    def run(self):
        try:
            for index, query in enumerate(self.workload):
                self.__queue.put(str(index) + "~#~" + query)
        except Exception as e:
            print(f"Producer error: {str(e)}")


def consumer_process(task_key):
    global total_lat
    global error_query_num
    
    query = task_key.split('~#~')[1]
    if query:
        try:
            start = time.time()
            result = mysql_query(query)
            end = time.time()
            interval = end - start

            if result:
                with lock:
                    total_lat += interval
            else:
                with lock:
                    error_query_num += 1
        except Exception as e:
            print(f"Process error: {str(e)}")
            with lock:
                error_query_num += 1


def run_job(thread_num=1, workload=[], resfile="../output.res"):
    global total_lat
    global error_query_num
    global POOL
    
    total_lat = 0
    error_query_num = 0
    workload_len = len(workload)
    start = time.time()

    # 限制线程数量，避免创建过多线程
    # 根据系统资源情况，设置一个合理的线程数上限
    max_threads = min(thread_num, 50)  # 限制最大线程数为50
    print(f"Using {max_threads} threads for processing")

    try:
        POOL = PooledDB(
            creator=pymysql,
            maxconnections=min(max_threads, 100),  # 限制最大连接数
            mincached=2,  # 保持最小连接数
            maxcached=5,  # 限制最大缓存连接数
            maxshared=3,
            blocking=True,
            maxusage=None,
            setsession=[],
            ping=1,  # 启用ping
            host=args["host"],
            port=int(args["port"]),
            user=args["user"],
            password=args["password"],
            database=args["database"],
            charset='utf8',
            connect_timeout=10  # 添加连接超时
        )

        # 使用有界队列，限制队列大小
        main_queue = queue.Queue(maxsize=1000)  # 限制队列大小
        p = Producer("Producer", main_queue, workload)
        p.start()

        # 使用线程池而不是直接创建线程
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # 创建进度条
            pbar = tqdm(total=workload_len, desc="Processing queries", unit="query")
            
            # 提交所有任务到线程池
            futures = []
            processed_count = 0
            
            # 使用批处理方式提交任务
            batch_size = 8000  # 每批处理的任务数
            current_batch = []
            
            while processed_count < workload_len:
                try:
                    # 非阻塞方式获取任务
                    task = main_queue.get_nowait()
                    current_batch.append(task)
                    processed_count += 1
                    main_queue.task_done()
                    
                    # 当批次达到指定大小或处理完所有任务时，提交批次
                    if len(current_batch) >= batch_size or processed_count >= workload_len:
                        for task_item in current_batch:
                            future = executor.submit(consumer_process, task_item)
                            future.add_done_callback(lambda p: pbar.update(1))
                            futures.append(future)
                        current_batch = []
                        
                        # 等待当前批次完成，避免创建过多线程
                        for future in futures[-len(current_batch):]:
                            future.result()
                            
                except queue.Empty:
                    # 如果队列为空但还没处理完所有任务，等待一小段时间
                    time.sleep(0.1)
                    continue
            
            # 等待所有任务完成
            for future in futures:
                future.result()
            
            # 关闭进度条
            pbar.close()

        run_time = round(time.time() - start, 1)
        run_query_num = workload_len - error_query_num
        
        if run_query_num == 0:
            avg_lat = 0
            avg_qps = 0
        else:
            avg_lat = total_lat / run_query_num
            avg_qps = run_query_num / run_time

        text = "\navg_qps(queries/s): \t{}\navg_lat(s): \t{}\n".format(round(avg_qps, 4), round(avg_lat, 4))
        with open(resfile, "w+") as f:
            f.write(text)
            f.close()

        print("run_job运行结束\n脚本总耗时:{}秒,sql执行总耗时:{}秒\n共有{}条数据，执行成功{}条\n{}".format(
            str(run_time), str(total_lat), str(workload_len), str(run_query_num), text))

        return round(avg_qps, 4), round(avg_lat, 4)

    except Exception as e:
        print(f"run_job error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0
    finally:
        if POOL:
            POOL.close()


def mysql_query(sql: str) -> bool:
    try:
        global POOL
        conn = POOL.connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.close()
        conn.commit()
        return True
    except Exception as error:
        print("mysql execute: " + str(error))
        return False
    finally:
        if 'conn' in locals():
            conn.close()
