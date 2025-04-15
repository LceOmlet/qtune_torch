#!/usr/bin/env python3
import os
import time
import sys

# 添加父目录到路径，以便能够导入父目录中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import parse_args, get_db_type
from run_job import run_job

def load_workload(file_path):
    """加载工作负载文件"""
    if not os.path.exists(file_path):
        print(f"错误：工作负载文件不存在: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    return queries

def main():
    """主函数，执行PostgreSQL工作负载测试"""
    args = parse_args()
    db_type = get_db_type()
    
    if db_type != 'postgresql':
        print("错误：当前配置不是PostgreSQL。请检查config.ini文件中的db_type设置。")
        sys.exit(1)
    
    print(f"开始PostgreSQL工作负载测试...")
    print(f"数据库主机: {args['host']}")
    print(f"数据库端口: {args['port']}")
    print(f"数据库名称: {args['database']}")
    print(f"使用线程数: {args['thread_num']}")
    print(f"工作负载文件: {args['workload_file_path']}")
    
    # 加载工作负载
    workload = load_workload(args['workload_file_path'])
    print(f"加载了 {len(workload)} 条SQL查询")
    
    # 重复执行工作负载以获取更多数据点
    expanded_workload = []
    for i in range(5):  # 可以根据需要调整重复次数
        expanded_workload.extend(workload)
    
    print(f"将执行 {len(expanded_workload)} 条SQL查询")
    
    # 执行工作负载
    start_time = time.time()
    avg_qps, avg_lat = run_job(
        thread_num=int(args['thread_num']),
        workload=expanded_workload,
        resfile="output_pg.res"
    )
    end_time = time.time()
    
    # 打印结果
    print(f"\n测试完成，总运行时间: {end_time - start_time:.2f} 秒")
    print(f"平均QPS: {avg_qps}")
    print(f"平均延迟: {avg_lat} 秒")
    
    # 提供下一步建议
    print("\n要运行连接测试，请使用:")
    print("python3 test_db_connection_pg.py")
    
    print("\n要修改配置并再次运行测试，请编辑config.ini文件")

if __name__ == "__main__":
    main() 