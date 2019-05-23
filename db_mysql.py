import os
import sys
import pymysql
#from sqlalchemy import create_engine
import pandas as pd


class dbMySQL:
    def __init__(self, host="198.218.40.1", user="bi_etl", passwd="transmit", port=5258, dbname="gbase", mode=1):

        self.host = host
        self.user = user
        self.passwd = passwd
        self.port = port
        self.dbname = dbname
        self.mode = mode
        self.open_db_orig()

    def open_db(self):
        self.open_db_orig()

    def open_db_orig(self):
        if self.mode == 0:
            curclass = pymysql.cursors.Cursor
        elif self.mode == 1:
            curclass = pymysql.cursors.DictCursor
        elif self.mode == 2:
            curclass = pymysql.cursors.SSCursor
        elif self.mode == 3:
            curclass = pymysql.cursors.SSDictCursor
        else:
            raise Exception("mode value is wrong")

        self.mycon = pymysql.connect(host=self.host,
                                     user=self.user,
                                     passwd=self.passwd,
                                     db=self.dbname, port=self.port, charset='utf8', cursorclass=curclass)
        self.cur = self.mycon.cursor()

    def open_cur(self):
        self.cur.close()
        self.cur = self.mycon.cursor()

    def iCommit(self):
        self.open_cur()
        self.mycon.commit()

    def closeDB(self):
        self.cur.close()
        self.mycon.close()


class MysqlEngine():

    def __init__(self):
        self.set_zkh_crm_para()

    def set_zkh_crm_para(self):

        self.host = """198.218.40.1"""
        self.user = """bi_etl"""
        self.passwd = """transmit"""
        self.port = 5258
        self.dbname = 'gbase'
        self.con = pymysql.connect(host=self.host,
                                   user=self.user,
                                   passwd=self.passwd,
                                   db=self.dbname, port=self.port, charset='utf8')

    def query_db(self, sql, index_col=None):
        df = pd.read_sql_query(sql, self.con, index_col)
        return df

    def read_db(self, tb_name):
        try:
            df = pd.read_sql_table(tb_name, self.con, schema=self.dbname)
        except Exception as e:
            print(e)
            df = None
        return df

    def save_db(self, df, tb_name):
        df.to_sql(tb_name, self.con, if_exists='append', chunksize=1000, schema=self.dbname, index=False)

    def close_db(self):
        self.con.close()


if __name__ == "__main__":
    db = dbMySQL()
    print('succeed:')
    data = pd.read_csv('C:/Users/Henry/Desktop/todb2.csv', index_col=0)

    query = """insert into rmdssdb.ttrms_vnp_aoh_train_label_2 (start_depart_date,end_depart_date,from_station_telecode,
    to_station_telecode,train_code,start_time_int,tra_time,cluster_label) 
       values ('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}')"""
    # mysql_enngine = MysqlEngine()
    for r in range(0, len(data)):
        print(r)
        start_depart_date = data.iloc[r, 0]

        end_depart_date = data.iloc[r, 1]
        from_station_telecode = data.iloc[r, 2]
        to_station_telecode = data.iloc[r, 3]
        train_code = data.iloc[r, 4]
        start_time_int = data.iloc[r, 5]
        tra_time = data.iloc[r, 6]
        cluster_label = data.iloc[r, 7]
        query_daily_result = db.cur.execute(query.format(start_depart_date, end_depart_date, from_station_telecode,
                                                           to_station_telecode, train_code,start_time_int, tra_time,
                                                         cluster_label))

    db.cur.close()

    db.iCommit()

    db.closeDB()

