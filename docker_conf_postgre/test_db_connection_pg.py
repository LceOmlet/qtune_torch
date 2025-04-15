#!/usr/bin/env python3
import sys
import os

# 添加父目录到路径，以便能够导入父目录中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
import psycopg2
from configs import parse_args, get_db_type

def test_connection():
    """Test the database connection based on configuration"""
    args = parse_args()
    db_type = get_db_type()
    
    print(f"Testing connection to {db_type} database...")
    print(f"Host: {args['host']}")
    print(f"Port: {args['port']}")
    print(f"Database: {args['database']}")
    print(f"User: {args['user']}")
    
    try:
        if db_type == 'mysql':
            conn = pymysql.connect(
                host=args["host"],
                user=args["user"],
                passwd=args["password"],
                port=int(args["port"]),
                database=args["database"],
                connect_timeout=30,
                charset='utf8'
            )
            print("MySQL connection successful!")
            
            # Test a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"MySQL Version: {version[0]}")
            
            # Get tables
            cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema='{args['database']}'")
            tables = cursor.fetchall()
            print(f"Tables in database: {[table[0] for table in tables]}")
            
            cursor.close()
            conn.close()
            
        elif db_type == 'postgresql':
            conn = psycopg2.connect(
                host=args["host"],
                user=args["user"],
                password=args["password"],
                port=int(args["port"]),
                database=args["database"],
                connect_timeout=30
            )
            print("PostgreSQL connection successful!")
            
            # Test a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            print(f"PostgreSQL Version: {version[0]}")
            
            # Get tables
            cursor.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'")
            tables = cursor.fetchall()
            print(f"Tables in database: {[table[0] for table in tables]}")
            
            cursor.close()
            conn.close()
            
        else:
            print(f"Unsupported database type: {db_type}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Connection error: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 