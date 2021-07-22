import csv
import numpy as np;
import matplotlib.pyplot as plt;

from config.settings import DATA_DIRS


class P230:
    def p248(self, loc):
        f = open(DATA_DIRS[0]+'\\age2.csv');
        data = csv.reader(f);
        next(data);
        data = list(data);

        home = None;
        home2 = None;
        # 신도림의 연령 별 비율
        for row in data:
            if loc in row[0]:
                home = np.array(row[3:],dtype=int);
                home2 = home / int(row[2].replace(',',''));
                #[0.2 0.3 0.4 .........]
        # 모든 지역의 연령 별 비율을 구한다.
        min = 999;
        result_name = '';
        result = None;
        for row in data:
            away = np.array(row[3:],dtype=int);
            away2 = away / int(row[2].replace(',',''));
            s = np.sum(np.abs(home2 - away2));
            if loc not in row[0] and s < min:
                min = s;
                result_name = row[0];
                result = away2;

        print(loc);
        print(home2.tolist());
        print(result_name.split(' ')[1]);
        print(result.tolist());
        result = [{
            'name': loc,
            'data': home2.tolist()
        },{
            'name': result_name.split(' ')[1],
            'data': result.tolist()
        }];
        print(result);
        return result;

    # 1. 서울시 영유아(5세이하)들이 가장 많이 사는 지역 구하시오
    def p2311(self):
        f = open(DATA_DIRS[0]+'\\age3.csv');
        data = csv.reader(f);
        next(data);
        data = list(data);
        result = [];
        for row in data:
            datas = [];
            baby = np.array(row[3:9],dtype=int);
            babys = int(np.sum(baby));
            # ['서초구',9080]
            datas.append(row[0].split(' ')[1]);
            datas.append(babys);
            result.append(datas);
        print(result);
        return result;

    def p231(self):
        f = open('age3.csv');
        data = csv.reader(f);
        next(data);
        data = list(data);
        max = 0;
        maxloc = '';
        maxrate = 0.0;
        min = 999999;
        minloc = '';
        minrate = 0.0;
        for row in data:
            rdata = np.array(row[3:9],dtype=int);
            sumdata = np.sum(rdata);
            if sumdata < min:
                minrate = sumdata / int(row[1]) * 100;
                min = sumdata;
                minloc = row[0];
            if sumdata > max:
                maxrate = sumdata / int(row[1]) * 100;
                max = sumdata;
                maxloc = row[0];
        print(maxloc, max, maxrate);
        print(minloc, min, minrate);


    # 2. 서울에서 5년간(18~20) 인구가 가능 많이 늘어난 구를 구하시오
    #    연령구분 10세  0~100세 데이터 다운로드(전체시군구현황)
    #    후 총 인구수로 계산
    def p232(self):
        # 2020
        f = open(DATA_DIRS[0]+'\\age4.csv');
        data = csv.reader(f);
        next(data);
        data = list(data);
        # 2015
        f2 = open(DATA_DIRS[0]+'\\age5.csv');
        data2 = csv.reader(f2);
        next(data2);
        data2 = list(data2);

        print(len(data));
        print(len(data2));

        locs = [];
        datas = [];

        for i in range(0,len(data)):
            sub = int(data[i][27]) - int(data2[i][1]);
            loc = data[i][0].split(' ')[1];
            datas.append(sub);
            locs.append(loc);
            #print(result, loc);


        result = {
            'locs':locs,
            'datas':datas
        };
        #print(result);
        return result;




        # max = 9999;
        # loc = '';
        # for row in data:
        #     rdata = (int(row[27]) - int(row[1]));
        #     if  rdata > max:
        #         max = rdata;
        #         loc = row[0];
        # print(loc, max);

if __name__ == '__main__':
    P230().p232();
