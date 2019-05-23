import pandas as pd
import ConnectToGbase
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

OD_UPNUM_TABLENAME = "rmdssdb.ttrms_upnum_from_statistics_result"
HOLIDAY_FLAG_TABLENAME="rmdssdb.d_period_holiday"

#OD_CODE=[['KNH','AOH'],['AOH','KNH'],['LJP','VNP'],['VNP','LJP'],['NKH','AOH'],['AOH','NKH'],['UUH','NKH'],['NKH','UUH'],['JGK','VNP'],['VNP','NKH'],['AOH','VNP'],['VNP','AOH']]
OD_CODE=[['VNP','AOH']]
class UpNumClassify:
    def __init__(self,lst_od_comb,str_start_date,str_end_date,split_n = 4,n_var = 1):
        self.gbase = ConnectToGbase.ConnetcToGbase()
        self.str_start_date = str_start_date
        self.str_end_date = str_end_date
        self.split_n = split_n
        self.n_var = n_var
        self.lst_od_comb = lst_od_comb
        self.__df_classify_result = None
        self.__col_names = ['start_depart_date','end_depart_date','from_station_telecode','to_station_telecode','train_no','start_time_int','tra_time','cluster_label']
        self.__default_cluster_n = 5
        
        
    def __grap_upnum_on_train_code(self,from_station_telecode,to_station_telecode,train_no):
        sql = f"select A.depart_date,sum(up_num) from "+OD_UPNUM_TABLENAME+" A ,"+HOLIDAY_FLAG_TABLENAME+" B "+f" where B.solar_day=A.depart_date and length(peak_name)<1 and train_no=\'{train_no}\' and  depart_date>=\'{self.str_start_date}\' and depart_date<=\'{self.str_end_date}\' and from_station_telecode=\'{from_station_telecode}\' and to_station_telecode=\'{to_station_telecode}\' group by A.depart_date order by A.depart_date"
        rows = self.gbase.getManyRows(sql)
        lst_up_num = [int(row[1]) for row in rows]
        return lst_up_num
    
    def __grap_distinct_train_code(self,from_station_telecode,to_station_telecode):
        sql = f"select distinct train_no,cast(substring(start_time,1,2) as int)+round(cast(substring(start_time,3,2) as int)/60,1) as start_time_int,tra_time from "+OD_UPNUM_TABLENAME+" A ,"+HOLIDAY_FLAG_TABLENAME+" B "+f" where B.solar_day=A.depart_date and tra_time is not null and length(peak_name)<1 and depart_date>=\'{self.str_start_date}\' and depart_date<=\'{self.str_end_date}\' and from_station_telecode=\'{from_station_telecode}\' and to_station_telecode=\'{to_station_telecode}\' "
        rows = self.gbase.getManyRows(sql)
        lst_train_detail = [[str(row[0]).strip(),float(row[1]),float(row[2])] for row in rows]
        print(lst_train_detail)
        return lst_train_detail
        
    def __split_upnum_range(self):
        lst_data_sources = []
        for i,val in enumerate(self.lst_od_comb):
            lst_train_detail = self.__grap_distinct_train_code(val[0],val[-1])
            lst_res = []
            lst_data_sources_tmp = []
            for train_detail in lst_train_detail:
                lst_up_num = self.__grap_upnum_on_train_code(val[0],val[-1],train_detail[0])
                if lst_up_num :
                    avg_val = np.average(lst_up_num)
                    var_val = np.var(lst_up_num)**0.5
                    lst_res.append([avg_val+self.n_var*var_val,avg_val-self.n_var*var_val])
                    lst_data_sources_tmp.append([self.str_start_date,self.str_end_date,val[0],val[-1],train_detail[0],train_detail[1],train_detail[2]])
            if len(lst_res) > self.__default_cluster_n:
                kmeans = KMeansAlg(lst_res,self.__default_cluster_n)
                best_labels = kmeans.KMAlg()
                for j,label_val in enumerate(best_labels):
                    lst_data_sources_tmp[j].append(label_val)
            else:
                continue
            lst_data_sources = lst_data_sources_tmp + lst_data_sources
        self.__df_classify_result = pd.DataFrame(lst_data_sources, columns=self.__col_names)
        
    
    @property
    def classify_result(self):
        self.__split_upnum_range()
        self.gbase.closeConnect()
        return self.__df_classify_result
            
class KMeansAlg:
    def __init__(self,label_frequency,cluster_num=5):
        self.label_frequency = np.array(label_frequency)
        self.max_n_cluster = cluster_num
        
    def KMAlg(self):
        k = 2
        best_ch = 0
        best_labels = []
        while k <= self.max_n_cluster:
            k_means = KMeans(init='k-means++',n_clusters=k,random_state=10)
            k_means.fit(self.label_frequency)
            k_means_labels = k_means.labels_
            ch = metrics.calinski_harabaz_score(self.label_frequency,k_means_labels)
            if ch > best_ch:
                best_ch = ch
                best_labels = list(k_means_labels)
            k = k + 1
        return best_labels

         
          
if __name__=="__main__":
    unc = UpNumClassify(OD_CODE, '20180410', '20190520')

    unc.classify_result.to_csv('C:/Users/Henry/Desktop/todb2.csv')