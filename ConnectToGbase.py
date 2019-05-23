import pymysql

HOST = '198.218.40.1'
USER = 'bi'
PASSWD = 'ytdxzcq'
DB ='rmdssdb'

class ConnetcToGbase:
    def __init__(self):
        self.conn = None
        self.cur = None
        self.conn = pymysql.connect(host='198.218.40.1',port=5258,user='bi',passwd='ytdxzcq',db='rmdssdb')
        self.cur = self.conn.cursor()
        self.conn.autocommit(1)
        
    def getManyRows(self,sql):
        self.cur.execute(sql)
        return self.cur.fetchall()
    
    def setHandleRows(self,sql):
        self.cur.execute(sql)
        
    def closeConnect(self):
        self.cur.close()
        self.conn.close()
    
    def execProcNoPara(self,proc_name):
        self.cur.callproc(proc_name)
        

