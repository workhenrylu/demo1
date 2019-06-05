import pandas as pd
import ConnectToGbase
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from datetime import date
import re
import db_mysql
#  对车次进行傅里叶变化试图寻找周期
class TableGet():

    def __init__(self, today, day1, odb, ode):
        self.Gbase = ConnectToGbase.ConnetcToGbase()
        self.today = today
        self.day1 = day1
        self.odb = odb
        self.ode = ode
        self.num1 = self.get_ave_num(self.day1, self.today)

    def get_ave_num(self, date_start, date_end):
        sql = f"select sum(up_num)/(" + date_end + "-" + date_start + ") as num from rmdssdb.ttrms_upnum_from_statistics_result " \
              f"where depart_date >= " + date_start + " and depart_date <= " + date_end + " and from_station_telecode = '"\
              + self.odb + "' and to_station_telecode = '" + self.ode + "'"

        rows = self.Gbase.getManyRows(sql)
        print('--------get ave_num_data successful----------')
        return float(re.findall(r"\d+\.?\d*", str(rows))[0])

    def get_nearest_table(self, date_start, date_end):
        sql = f"select A.current_train_code,date_format(A.depart_date,'%Y%m%d') as depart_date,sum(up_num) as up_num,cast(substring(start_time,1,2) as int)+" \
                  f"round(cast(substring(start_time,3,2) as int)/60,1) as start_time_int,tra_time," \
                  f"from_station_telecode,to_station_telecode " \
                  f"from rmdssdb.beijin_shanghai_test A , rmdssdb.d_period_holiday B " \
                  f"where B.solar_day=date_format(A.depart_date,'%Y%m%d') and length(peak_name)<1 " \
                  f"and depart_date >= '" + date_start[0:4]+'-'+date_start[4:6]+'-'+date_start[6:8] + "' and depart_date <= '" + date_end[0:4]+'-'+date_end[4:6]+'-'+date_end[6:8] + "' and from_station_telecode != to_station_telecode " \
                  f"and from_station_telecode = '" + self.odb + f"' and to_station_telecode = '" + self.ode + \
                  f"' group by current_train_code,depart_date,A.start_time,A.tra_time,A.from_station_telecode,A.to_station_telecode"
        rows = self.Gbase.getManyRows(sql)
        basic_table = pd.DataFrame(list(rows),
                                   columns=['train_code', 'depart_date', 'up_num', 'start_time_int', 'tra_time',
                                            'from_station_telecode', 'to_station_telecode'])
        print('get neareat table successful')
        train_lst = [i.strip() for i in list(basic_table['train_code'].unique())]
        return (basic_table, train_lst)
        #返回值是一个dataframe和一个list

    def get_gamma(self, date_start, date_end):

        date_start=date_start
        date_end=date_end
        num2 = self.get_ave_num(date_start, date_end)
        print('get GAMMA')
        return self.num1/num2

    def get_old(self, date_start, date_end, gamma, train_lst):
        sql = f"select A.current_train_code,date_format(A.depart_date,'%Y%m%d') as depart_date,sum(up_num)* "+str(gamma)+"as up_num,cast(substring(start_time,1,2) as int)+" \
                  f"round(cast(substring(start_time,3,2) as int)/60,1) as start_time_int,tra_time," \
                  f"from_station_telecode,to_station_telecode " \
                  f"from rmdssdb.beijin_shanghai_test A , rmdssdb.d_period_holiday B " \
                  f"where B.solar_day=date_format(A.depart_date,'%Y%m%d') and length(peak_name)<1 " \
                  f"and tra_time is not null and current_train_code not in (" + str(train_lst).replace('[', '').replace(']',
                                                                                                                '') + \
              f") and depart_date >= '" + date_start[0:4]+'-'+date_start[4:6]+'-'+date_start[6:8]+ "' and depart_date <= '" +date_end[0:4]+'-'+date_end[4:6]+'-'+date_end[6:8]+ "' and from_station_telecode != to_station_telecode " \
                  f"and from_station_telecode = '" + self.odb + f"' and to_station_telecode = '" + self.ode + \
              f"' group by current_train_code,depart_date,A.start_time,A.tra_time,A.from_station_telecode,A.to_station_telecode"
        rows = self.Gbase.getManyRows(sql)
        basic_table = pd.DataFrame(list(rows),
                                   columns=['train_code', 'depart_date', 'up_num', 'start_time_int', 'tra_time',
                                            'from_station_telecode', 'to_station_telecode'])
        print('get old', basic_table['train_code'].unique())
        return basic_table

#  出结果


class Find():

    def __init__(self):
        self.Gbase2 = ConnectToGbase.ConnetcToGbase()

    def __get_date_lst(self,odb,ode):
        sql = f"select date_format(start_date,'%Y%m%d') as date from rmdssdb.od_date_split " \
              f"where from_station_telecode = '"+odb+"' and to_station_telecode = '"+ode+"' and start_date >='2016-01-01' order by date desc"
        rows = self.Gbase2.getManyRows(sql)
        print('--------get data list successful----------')
        lst = [list(i)[0] for i in rows]
        if len(lst) >= 1:
            return lst
        else:
            return ['20160101', '20190511']

    def get_totallist(self, ood_code):
        od_code = ood_code
        #today = str(date.today()).replace("-", "")
        date_list = []
        for n in range(0, len(od_code)):
            date_list_tmp = Find().__get_date_lst(od_code[n][0], od_code[n][1])
            date_list = date_list + date_list_tmp
            date_list = list(set(date_list))

        date_list.sort(reverse=True)
        date_list = [today] + date_list
        print('get total list',date_list)
        return date_list

    def to_you_table(self, od_code):
        ood_code = od_code
        date_list = self.get_totallist(ood_code)
        #  得到total时间切割

        tb = pd.DataFrame()
        for m in range(0, len(ood_code)):

            tb_temp = TableGet(date_list[0], date_list[1], ood_code[m][0], ood_code[m][1])
            tb_lst_combine = tb_temp.get_nearest_table(date_list[1], date_list[0])
            table = pd.DataFrame(tb_lst_combine[0])
            if len(date_list) > 2:
                if len(table) > 0:
                    train_lst = tb_lst_combine[1]

                    for i in range(1, len(date_list) - 1):
                        gma = tb_temp.get_gamma(date_list[i + 1], date_list[i])
                        old_table_tmp = pd.DataFrame(tb_temp.get_old(date_list[i + 1], date_list[i], gma, train_lst))
                        table = pd.concat([table, old_table_tmp])
            tb = pd.concat([tb, table])
            print('od already')

        return processing(tb)

def fly(up_num):
    if len(up_num) > 1:
        return len(up_num)/np.argmax(np.fft.fft(up_num)[1:int(len(up_num)/2)])
    else:
        return 0

def processing(basic_table):
    #  取出的数据格式不对，要调整一下
    basic_table['up_num'] = basic_table['up_num'].astype(int)
    basic_table['start_time_int'] = basic_table['start_time_int'].astype(int)
    basic_table['tra_time'] = basic_table['tra_time'].astype(int)
    basic_table['depart_date'] = pd.to_datetime(basic_table['depart_date'], format='%Y%m%d')

    #  拼凑一个暂时的中间表，一条车次留一条信息，取出做出标准差，均值，μ+σ，μ-σ
    final_table = basic_table.groupby(
        ['train_code', 'from_station_telecode', 'to_station_telecode']).mean().reset_index()

    temp_tb = pd.DataFrame(columns=['train_code', 'period'])
    for tc in basic_table['train_code'].unique():
        x = basic_table[basic_table['train_code'] == tc]['up_num']
        ss = pd.DataFrame({'train_code': tc, 'period': fly(x)}, index=[0])
        temp_tb = pd.concat([temp_tb, ss])
    final_table = pd.merge(final_table, temp_tb, on='train_code')

    final_table = pd.merge(final_table,
                           basic_table.groupby(['train_code', 'from_station_telecode', 'to_station_telecode']) \
                               ['up_num'].std().reset_index(),
                           on=['train_code', 'from_station_telecode', 'to_station_telecode'])
    final_table = final_table.rename(columns={'up_num_x': 'mean', 'up_num_y': 'std'})
    final_table['top'] = final_table['mean'] + final_table['std']
    final_table['bottom'] = final_table['mean'] - final_table['std']
    final_table = final_table.replace(np.nan, 0)
    final_table = final_table.replace(np.inf, 0)

    if len(final_table) > 2:
        #  控制传入车次的数量一定要大于2，否则后面的kmeans无法用
        #  minmax简单对数据进行处理
        final_table['mean'] = (final_table['mean'] - min(final_table['mean'])) / (
                    max(final_table['mean']) - min(final_table['mean']))
        final_table['std'] = (final_table['std'] - min(final_table['std'])) / (
                    max(final_table['std']) - min(final_table['std']))
        final_table['top'] = (final_table['top'] - min(final_table['top'])) / (
                    max(final_table['top']) - min(final_table['top']))
        final_table['bottom'] = (final_table['bottom'] - min(final_table['bottom'])) / (
                    max(final_table['bottom']) - min(final_table['bottom']))
        final_table['period'] = (final_table['period'] - min(final_table['period'])) / (
                    max(final_table['period']) - min(final_table['period']))
        #  中间表取出相关数据做array
        arr = np.array(final_table[['std', 'top', 'bottom', 'period']])
        kmeans = KMeansAlg(arr, 5)
        best_labels = kmeans.KMAlg()
        final_table['cluster_label'] = best_labels
    else:
        #  强行label为99
        final_table['cluster_label'] = 99

    #  调整最后的输出表
    result = final_table[['train_code', 'start_time_int', 'tra_time', 'from_station_telecode',
                          'to_station_telecode', 'cluster_label']].copy()
    result['start_depart_date'] = '20160101'
    result['end_depart_date'] = today
    result = result[
        ['start_depart_date', 'end_depart_date', 'from_station_telecode', 'to_station_telecode', 'train_code',
         'start_time_int', 'tra_time', 'cluster_label']]

    # result = result[
    #      ['train_code', 'start_time_int', 'tra_time', 'cluster_label']]
    return result


class KMeansAlg:
    def __init__(self, label_frequency, cluster_num=5):
        self.label_frequency = np.array(label_frequency)
        self.max_n_cluster = cluster_num

    def KMAlg(self):
        k = 3
        best_ch = 0
        best_labels = []
        while k <= self.max_n_cluster:
            k_means = KMeans(init='k-means++', n_clusters=k, random_state=10)
            k_means.fit(self.label_frequency)
            k_means_labels = k_means.labels_
            ch = metrics.calinski_harabasz_score(self.label_frequency, k_means_labels)
            if ch > best_ch:
                best_ch = ch
                best_labels = list(k_means_labels)
            k = k + 1
        return best_labels


if __name__=='__main__':
    today = str(date.today()).replace("-", "")

    ooood_code = [['AOH','VNP']]
    #  返回dataframe
    #print(Find().to_you_table(ooood_code))





    #  写库
    # db = db_mysql.dbMySQL()
    # print('succeed:')
    # data = Find().to_you_table(ooood_code)
    #
    # query = """insert into rmdssdb.ttrms_sh_bj_cluster (start_depart_date,end_depart_date,from_station_telecode,
    #     to_station_telecode,train_code,start_time_int,tra_time,cluster_label)
    #        values ('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}')"""
    # # mysql_enngine = MysqlEngine()
    # for r in range(0, len(data)):
    #     print(r)
    #     start_depart_date = data.iloc[r, 0]
    #
    #     end_depart_date = data.iloc[r, 1]
    #     from_station_telecode = data.iloc[r, 2]
    #     to_station_telecode = data.iloc[r, 3]
    #     train_code = data.iloc[r, 4]
    #     start_time_int = data.iloc[r, 5]
    #     tra_time = data.iloc[r, 6]
    #     cluster_label = data.iloc[r, 7]
    #     query_daily_result = db.cur.execute(query.format(start_depart_date, end_depart_date, from_station_telecode,
    #                                                      to_station_telecode, train_code, start_time_int, tra_time,
    #                                                      cluster_label))
    #
    # db.cur.close()
    #
    # db.iCommit()
    #
    # db.closeDB()

