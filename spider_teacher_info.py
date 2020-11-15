import requests
import bs4
import json
from lxml import etree
import pprint

def get_url_list():
    url = "http://www.caitec.org.cn/n4/sy_byxz_yjxlzj/"
    r = requests.get(url)
    r.encoding = 'utf-8'
    demo = r.text
    soup = bs4.BeautifulSoup(demo,'html.parser')
    lst = []
    for k in soup.dd.parent.find_all('a'):
        lst.append(k['href'])
    
    del lst[1]
    
    lst2=[]
    for url_1 in lst:
        for i in range(1,7):
            url = 'http://www.caitec.org.cn' + url_1 + 'json/index_'+str(i)+'.html'
            lst2.append(url)
    
    lst3=[]
    for url in lst2:
        r = requests.get(url)
        r.encoding = 'utf-8'
        demo = r.text
        if len(demo)> 210:
            json_str = json.loads(demo) 
            a = json_str['data']
            for k in a:
                lst3.append("http://www.caitec.org.cn"+k['alink'])
 
    return lst3

if __name__ == "__main__":
    lst3=get_url_list()


    # 日常爬去需要解禁lst
    for url in lst3[:]:
        r = requests.get(url)
        r.encoding='utf-8'
        html = etree.HTML(r.text)
        result1 = html.xpath("//div[@class='scholar-head-info']/*//text()")
        result2 = html.xpath("//div[@class='scholar-art-bd']/*//text()")

        # print(result1)
        pprint.pprint([' '.join([i.strip() for i in price.strip().split('\t')]) for price in result1])
        print("---------")
        # print(result2)
        if len(result2) == 0:
            pprint.pprint('该老师并未完善此部分')
        else:
            pprint.pprint([' '.join([i.strip() for i in price.strip().split('\t')]) for price in result2])
        print("\n\n")
