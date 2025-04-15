import numpy as np
import pandas
import json
import os
import pymysql
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from configs import predictor_output_dim, get_db_type

query_types = ["insert", "delete", "update", "select"]


# base prediction model
class BaselineModel(nn.Module):
    def __init__(self, num_feature=len(query_types), output_dim=predictor_output_dim):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(num_feature, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def baseline_model(num_feature=len(query_types)):
    # create model
    model = BaselineModel(num_feature=num_feature)
    # Note: In PyTorch, we don't compile the model like in Keras
    # The loss function and optimizer are defined separately when training
    return model


class SqlParser:
    ###########################################################################
    # DML: select delete insert update      0 1 2 3
    # select {select_expr}
    # <modifier> (The first is default)
    # [ALL | DISTINCT | DISTINCTROW]
    # [0 | HIGH_PRIORITY], faster than update, with table-level lock
    # [0 | STRAIGHT_JOIN],
    # [0 | SQL_SMALL_RESULT | SQL_BIG_RESULT]
    # [0 | SQL_BUFFER_RESULT]
    # [SQL_CACHE | SQL_NO_CACHE]
    # [SQL_CALC_FOUND_ROWS]

    # OLTP (workload2vector)
    # select*w1 + sum(modifiers)*w2 + num({select_expr})*wl3        # 0.7 0.1 0.2
    # from [table]
    # [WHERE where_condition]   range join

    # OLTP (sql2vector)
    # cost-vector: [Aggregate, Nested Loop, Index Scan, Hash_Join]

    # Keywords
    # [GROUP BY {col_name | expr | position}]
    # [ASC | DESC], ...[WITH ROLLUP]
    # [HAVING where_condition]
    # [ORDER BY {col_name | expr | position}]
    # [ASC | DESC], ...

    # sum(group_table_scale(having)*wi) + order_cost*wi
    ###########################################################################

    def __init__(self, argus):

        self.resfile = os.path.join("scripts/") + "output.res"
        self.argus = argus
        self.db_type = get_db_type()
        self.conn = self.db_conn()
        self.tables = self.get_database_tables()
        self.query_encoding_map = {}
        ########### Convert from the sql statement to the sql vector
        #  directly read vector from a file (so a python2 script needs to run first!)
        #  sql_type * (num_events, C, aggregation, in-mem)
        #############################################################################################################################

        # query encoding features
        self.op_weight = {'oltp_point_select': 1, 'select_random_ranges': 2, 'oltp_delete': 3,
                          'oltp_insert': 4, 'bulk_insert': 5, 'oltp_update_index': 6,
                          'oltp_update_non_index': 7, }
        self.num_event = int(argus['num_event'])
        self.C = [10000]
        self.group_cost = 0
        self.in_mem = 0
        self.predict_sql_resource_value = None
        self.estimator = baseline_model(len(query_types) + len(self.tables))
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator.to(self.device)
        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.estimator.parameters())
        self.criterion = nn.MSELoss()
        
        # Prepare Data
        fs = open("training-data/trainData_sql.txt", 'r')
        df = pandas.read_csv(fs, sep=' ', header=None)
        lt_sql = df.values
        # seperate into input X and output Y
        sql_op = lt_sql[:, 0]
        sql_X = lt_sql[:, 1:5]  # op_type   events  table_size
        sql_Y = lt_sql[:, 5:]

    def query_encoding(self, query):

        if not query:
            print("query is empty")
            return []

        if self.query_encoding_map.get(str(query), None):
            return self.query_encoding_map[str(query)]

        result = [0 for i in range(len(self.tables) + len(query_types))]
        # [0, 0, 0, 0, X, X, X..........]
        query_split_list = query.lower().split(" ")

        for index, query_type in enumerate(query_types):
            if query_type in query_split_list:
                result[index] = 1

        query_split_list = query.replace(",", "").split(" ")

        if self.db_type == 'mysql':
            explain_format_fetchall = self.db_query("EXPLAIN FORMAT=JSON {};".format(query))
            if not explain_format_fetchall:
                print("explain_format_fetchall is empty, query: {}".format(query))
                return []
            explain_format = json.loads(explain_format_fetchall[0][0])
            explain_format_tables_list = self.get_explain_format_tables_list([], explain_format.get("query_block"), "table")
            for explain_format_table in explain_format_tables_list:
                explain_format_table_name = explain_format_table["table_name"]
                index = query_split_list.index(explain_format_table_name)
                if query_split_list[index - 1].lower() == "as":
                    explain_format_table_name = query_split_list[index - 2]
                else:
                    explain_format_table_name = query_split_list[index - 1]

                for index, table_name in enumerate(self.tables):
                    if explain_format_table_name == table_name:
                        result[index + len(query_types)] = float(explain_format_table["cost_info"]["prefix_cost"])
                        continue
                        
        elif self.db_type == 'postgresql':
            # PostgreSQL uses a different EXPLAIN format
            explain_fetchall = self.db_query("EXPLAIN (FORMAT JSON) {};".format(query))
            if not explain_fetchall or not explain_fetchall[0][0]:
                print("explain_fetchall is empty, query: {}".format(query))
                return []
                
            explain_data = explain_fetchall[0][0]
            # Extract table names and costs from PostgreSQL's explain output
            table_costs = self.extract_pg_table_costs(explain_data, query_split_list)
            
            for table_name, cost in table_costs:
                for index, known_table in enumerate(self.tables):
                    if table_name == known_table:
                        result[index + len(query_types)] = float(cost)
                        continue
                        
        self.query_encoding_map[str(query)] = result
        return result

    def extract_pg_table_costs(self, explain_data, query_split):
        """Extract table names and costs from PostgreSQL's explain JSON output"""
        table_costs = []
        
        # Parse the JSON explain output
        try:
            plan_data = json.loads(explain_data)[0]['Plan']
            self._extract_tables_from_plan(plan_data, table_costs)
        except Exception as e:
            print(f"Error parsing PostgreSQL EXPLAIN output: {str(e)}")
        
        return table_costs
    
    def _extract_tables_from_plan(self, plan, table_costs, parent_cost=0):
        """Recursively extract table information from PostgreSQL explain plan"""
        if 'Relation Name' in plan:
            table_name = plan['Relation Name']
            # Use Total Cost or the parent's cost
            cost = plan.get('Total Cost', parent_cost)
            table_costs.append((table_name, cost))
            
        # Process child plans
        if 'Plans' in plan:
            for child_plan in plan['Plans']:
                self._extract_tables_from_plan(child_plan, table_costs, plan.get('Total Cost', 0))

    def predict_sql_resource(self, workload=[]):
        # Predict sql convert
        # inner_metric_change   np.array
        if self.predict_sql_resource_value is None:
            print("predict_sql_resource_value is None")
            exit()
        return self.predict_sql_resource_value
        # PyTorch version of prediction
        # workload_encoding = self.get_workload_encoding(workload)
        # workload_tensor = torch.FloatTensor(workload_encoding).to(self.device)
        # with torch.no_grad():
        #     prediction = self.estimator(workload_tensor)
        # return prediction.cpu().numpy()

    def update(self):
        pass

    def db_conn(self):
        """Create a database connection based on the configured type"""
        if self.db_type == 'mysql':
            return self.mysql_conn()
        elif self.db_type == 'postgresql':
            return self.postgresql_conn()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def mysql_conn(self):
        conn = pymysql.connect(
            host=self.argus["host"],
            user=self.argus["user"],
            passwd=self.argus["password"],
            port=int(self.argus["port"]),
            connect_timeout=30,
            charset='utf8')
        conn.select_db(self.argus["database"])
        return conn
        
    def postgresql_conn(self):
        conn = psycopg2.connect(
            host=self.argus["host"],
            user=self.argus["user"],
            password=self.argus["password"],
            port=int(self.argus["port"]),
            database=self.argus["database"],
            connect_timeout=30)
        return conn

    def close_db_conn(self):
        try:
            self.conn.close()
        except Exception as error:
            print("close db connection: " + str(error))

    def db_query(self, sql):
        """Execute a query based on the configured database type"""
        if self.db_type == 'mysql':
            return self.mysql_query(sql)
        elif self.db_type == 'postgresql':
            return self.postgresql_query(sql)
        else:
            print(f"Unsupported database type: {self.db_type}")
            return None

    def mysql_query(self, sql):
        try:
            cursor = self.conn.cursor()
            count = cursor.execute(sql)
            if count == 0:
                result = 0
            else:
                result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as error:
            print("MySQL execute: " + str(error))
            return None
            
    def postgresql_query(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            try:
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                # No results to fetch
                result = 0
            cursor.close()
            return result
        except Exception as error:
            print("PostgreSQL execute: " + str(error))
            return None

    def get_database_tables(self):
        # get all tables
        if self.db_type == 'mysql':
            tables_fetchall = self.db_query(
                "select table_name from information_schema.tables where table_schema='{}';".format(self.argus["database"]))
        elif self.db_type == 'postgresql':
            tables_fetchall = self.db_query(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';")
        else:
            print(f"Unsupported database type: {self.db_type}")
            return []
            
        tables = []
        if not tables_fetchall:
            print("tables was not found")
            return []
        for table in tables_fetchall:
            if table and table[0]:
                tables.append(table[0])
        print("get all tables success")
        return tables

    def get_explain_format_tables_list(self, values_list, json, key):
        if isinstance(json, dict):
            for item, values in json.items():
                if str(item) == str(key):
                    values_list.append(json.get(item))
                if isinstance(values, dict):
                    self.get_explain_format_tables_list(values_list, values, key=key)
                if isinstance(values, list):
                    self.get_explain_format_tables_list(values_list, values, key=key)
                else:
                    pass
        elif isinstance(json, list):
            for data in json:
                if isinstance(data, dict):
                    self.get_explain_format_tables_list(values_list, data, key)
        else:
            return values_list
        return values_list

    def get_workload_encoding(self, workload):
        queries_encoding = []
        for query in workload:
            queries_encoding.append(self.query_encoding(query))

        # [0, 0, 0, 0, X, X, X..........]
        workload_encoding = np.array([0 for i in range(len(self.tables) + len(query_types))])
        for query_encoding in queries_encoding:
            workload_encoding = workload_encoding + np.array(query_encoding)

        for i in range(len(query_types)):
            if workload_encoding[i] > 0:
                workload_encoding[i] = 1

        return workload_encoding.reshape(1, len(workload_encoding))
        
    def fit(self, X, Y, batch_size=50, epochs=10):
        """
        PyTorch version of the fit method to replace Keras' fit
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_Y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.estimator(batch_X)
                loss = self.criterion(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")
            
    def save_weights(self, filepath):
        """
        PyTorch version of save_weights to replace Keras' save_weights
        """
        torch.save(self.estimator.state_dict(), filepath)
        
    def load_weights(self, filepath):
        """
        PyTorch version of load_weights to replace Keras' load_weights
        """
        self.estimator.load_state_dict(torch.load(filepath))
