#!/bin/bash
set -e

# 本地环境变量
export PATH=$PATH:/usr/bin:/usr/local/bin
PG_HOST="localhost"  # 连接到本地映射的Docker端口
PG_PORT="54321"      # 修正为配置文件中映射的正确端口
PG_USER="root"
PG_PASSWORD="123456"
PG_DB="sysbench"
table_size=8000000
table_num=20
# 等待PostgreSQL启动
echo "等待PostgreSQL服务就绪..."
MAX_RETRY=30
RETRY_INTERVAL=5
retry_count=0

while [ $retry_count -lt $MAX_RETRY ]; do
    if pg_isready -h $PG_HOST -p $PG_PORT -U $PG_USER; then
        echo "PostgreSQL服务已就绪！"
        break
    else
        echo "PostgreSQL尚未就绪，继续等待... (尝试 $retry_count/$MAX_RETRY)"
        sleep $RETRY_INTERVAL
        retry_count=$((retry_count + 1))
    fi
done

if [ $retry_count -ge $MAX_RETRY ]; then
    echo "错误: 等待PostgreSQL就绪超时"
    exit 1
fi

# 尝试创建sysbench测试表，失败时重试
echo "创建sysbench测试表..."
MAX_TABLE_RETRY=3
table_retry=0

# 直接删除已经存在的sysbench测试表
echo "删除现有的测试表(如果存在)..."
PGPASSWORD=$PG_PASSWORD psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d $PG_DB -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;" 2>/dev/null || true
echo "已重置数据库schema"

# 删除sysbench测试表    
drop_tables() {
    sysbench \
      --db-driver=pgsql \
      --pgsql-host=$PG_HOST \
      --pgsql-port=$PG_PORT \
      --pgsql-user=$PG_USER \
      --pgsql-password=$PG_PASSWORD \
      --pgsql-db=$PG_DB \
      /usr/share/sysbench/oltp_read_write.lua cleanup
}

# 创建sysbench测试表    
create_tables() {
    sysbench \
      --db-driver=pgsql \
      --pgsql-host=$PG_HOST \
      --pgsql-port=$PG_PORT \
      --pgsql-user=$PG_USER \
      --pgsql-password=$PG_PASSWORD \
      --pgsql-db=$PG_DB \
      --table-size=$table_size \
      --tables=$table_num \
      --threads=4 \
      /usr/share/sysbench/oltp_read_write.lua prepare
}

# 先尝试删除现有表
drop_tables
echo "尝试使用sysbench删除现有表..."

while [ $table_retry -lt $MAX_TABLE_RETRY ]; do
    if create_tables; then
        echo "sysbench测试表创建成功！"
        break
    else
        echo "创建测试表失败，正在重试... (尝试 $table_retry/$MAX_TABLE_RETRY)"
        sleep 5
        table_retry=$((table_retry + 1))
    fi
done

if [ $table_retry -ge $MAX_TABLE_RETRY ]; then
    echo "警告: 无法创建sysbench测试表，但将继续执行后续步骤"
fi


echo "=== 环境准备完成！ ==="
echo "请使用以下命令测试PostgreSQL连接："
echo "python3 docker_conf_postgre/test_db_connection_pg.py"
echo "请使用以下命令运行sysbench负载测试："
echo "python3 docker_conf_postgre/run_test_pg.py"