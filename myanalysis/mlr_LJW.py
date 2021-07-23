import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from config.settings import DATA_DIRS

# ------------------------------------------------------------------------------------------------------

class MLR:
    def healthPreprocessing(self, y):
        # 2009 ~ 2018년 건강 데이터
        dfh_all = pd.read_excel(DATA_DIRS[0] + '//health_2009_2018_preprocessed.xlsx', sheet_name = None, engine = 'openpyxl');
        dfh = pd.concat(dfh_all, ignore_index=True);

        # 분석 대상 지역 데이터 추출
        area = dfh['지역'].isin(['서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시','울산광역시','경기도','강원도','충청북도','충청남도','전라북도','전라남도','경상북도','경상남도','제주특별자치도'])
        dfh = dfh[area]

        # 분석 대상 건강지표 컬럼 추출
        health = dfh[['시도'] + y ]
        health = health.set_index('시도')

        # 건강 데이터 전처리 데이터프레임: health
        return health

    def waterPreprocessing(self, x):
        # 2009 ~ 2018년 수질 데이터
        dfw_all = pd.read_excel(DATA_DIRS[0] + '//water_2009_2018_preprocessed.xlsx', sheet_name = None, engine='openpyxl');
        dfw = pd.concat(dfw_all, ignore_index=True);

        # 필요없는 컬럼 drop
        dfw.drop(['시설명', '소재지', '수원', '채수년월일'], axis=1, inplace=True)

        # '납(기준:0.05/ 단위:(mg/L))', '비소(기준:0.05/ 단위:(mg/L))', '망간(기준:0.3/ 단위:(mg/L))'은 컬럼명이 달라 엑셀에서 컬럼명을 통일
        #'1,4-다이옥산(기준:0.05/ 단위:(mg/L))'은 2010년 부터, '브롬산염(기준:0.01/ 단위:(mg/L))'는 2017년 부터, '포름알데히드(기준:0.5/ 단위:(mg/L))'는 2014년 부터 측정하여 컬럼 제거
        dfw = dfw.drop(['1,4-다이옥산(기준:0.05/ 단위:(mg/L))', '브롬산염(기준:0.01/ 단위:(mg/L))', '포름알데히드(기준:0.5/ 단위:(mg/L))'], axis=1)

        # '총대장균군(기준:0/ 단위:MPN)', '대장균/분원성대장균군(기준:0/ 단위:MPN)' 컬럼에 '불검출' 개수가 많아 컬럼 삭제
        dfw = dfw.drop(['총대장균군(기준:0/ 단위:MPN)', '대장균/분원성대장균군(기준:0/ 단위:MPN)'], axis=1)

        # '냄새(기준:0/ 단위:(mg/L))', '맛(기준:0/ 단위:(mg/L))' 컬럼은 모든 값이 '적합'이므로 컬럼 삭제
        dfw = dfw.drop(['냄새(기준:0/ 단위:(mg/L))', '맛(기준:0/ 단위:(mg/L))'], axis=1)

        # 탁도 단위(NTU)의 호환성 때문에 컬럼 drop
        dfw.drop('탁도(기준:0.5/ 단위:(NTU))', axis=1, inplace=True)

        # 결측치를 가진 행 삭제
        dfw = dfw.dropna(axis=0)

        # 모든 값이 0인 컬럼 삭제
        dfw.replace(0, np.nan, inplace=True)                   # 0 -> NaN 변경
        nan_num = dfw.isnull().sum()                           # 컬럼별 결측치 수
        drop_list = list(nan_num[nan_num == len(dfw)].index)   # 모든 값이 0인 컬럼의 컬럼명 list
        water = dfw.drop(drop_list, axis=1)                    # 모든 값이 0인 컬럼 제거
        water.fillna(0, inplace=True)                          # 남은 NaN -> 0 되돌리기

        # 컬럼 명 변경
        water.rename({'수도사업자':'지역'}, axis=1, inplace=True)

        # ['검사월']을 연도만 남기고 ['year']로 컬럼명 변경
        water['검사월'] = water['검사월'].str[0:4]
        water.rename({'검사월':'year'}, axis=1, inplace=True)

        water = water[['지역','year','시설용량(㎥/일)'] + x ]

        # 수질 데이터 전처리 데이터프레임: water
        return water

    def waterQualMean(self):            # 각 지역별, 연도별 1년 평균 물질 농도를 계산하여 dataframe을 반환하는 함수

        lst = []   # Dataframe 만들기 위해서 준비
        years = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016','2017', '2018']
        cities = ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도']
        concs = water.columns[3:]

        for y in years:
            year = water['year'] == y
            water_y = water[year]
            for city in cities:
                ct_conc = [city + y]        # 하나의 도시에 대해, 이름과 모든 물질의 농도를 모은 리스트
                for conc in concs:
                    ct_water = water_y[water_y['지역'].str.contains(city)]
                    if conc == '일반세균(기준:100/ 단위:(CFU/mL))':
                        ct_water['월별물질농도(CFU/일)'] = ct_water['시설용량(㎥/일)'] * ct_water[conc] * (10**6)
                        t_conc = ct_water['월별물질농도(CFU/일)'].sum() / ct_water['시설용량(㎥/일)'].sum() * (10**-6)  # 2018년, xx지역, 일반세균 평균 농도(CFU/mL)

                    elif conc == '수소이온농도(기준:5.8 ~ 8.5/ 단위:-)':
                        ct_water[conc] = ct_water[conc].apply(lambda x: 1/ 10**x)   # pH를 [H+](단위:mol/L)로 바꾸는 과정

                        ct_water['월별물질농도(mol/일)'] = ct_water['시설용량(㎥/일)'] * ct_water[conc] * 1000
                        t_conc = ct_water['월별물질농도(mol/일)'].sum() / ct_water['시설용량(㎥/일)'].sum() * 0.001   # 2018년, xx지역, 수소이온 평균 농도(mol/L)
                        t_conc = -np.log10(t_conc)  # 원래대로 pH로 변환

                    else:    # '색도'포함
                        ct_water['월별물질농도(mg/일)'] = ct_water['시설용량(㎥/일)'] * ct_water[conc] * 1000
                        t_conc = ct_water['월별물질농도(mg/일)'].sum() / ct_water['시설용량(㎥/일)'].sum() * 0.001   # 2018년, xx지역, xx 물질 평균 농도(mg/L)
                    ct_conc.append(t_conc)
                lst.append(ct_conc)


        result = pd.DataFrame(lst, columns=['지역'] + list(concs))
        result = result.set_index('지역').drop(['세종특별자치시2009','세종특별자치시2010','세종특별자치시2011','세종특별자치시2012','세종특별자치시2013','세종특별자치시2014','세종특별자치시2015','세종특별자치시2016','세종특별자치시2017','세종특별자치시2018'], axis=0)
        return result

    def mlr(self, x, y):
        # Standardization
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # train, test data 나누기
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        # lr_coef = 기울기, lr_intercept = 절편
        lr_coef = lr.coef_
        lr_intercept = lr.intercept_

        # Test
        pd.DataFrame(lr.predict(x_test))
        score = lr.score(x_test, y_test)

        return score, lr_coef, lr_intercept

    # def visualization(self):

if __name__ == '__main__':
    MLR = MLR()

    # 독립변수, 종속변수 리스트 생성
    x_lst1 = ['증발잔류물(기준:500/ 단위:(mg/L))', '알루미늄(기준:0.2/ 단위:(mg/L))', '색도(기준:5/ 단위:(도))', '디브로모아세토니트릴(기준:0.1/ 단위:(mg/L))', '과망간산칼륨소비량(기준:10/ 단위:(mg/L))', '염소이온(기준:250/ 단위:(mg/L))', '불소(기준:1.5/ 단위:(mg/L))', '디브로모클로로메탄(기준:0.1/ 단위:(mg/L))']
    y_lst1 = ['스트레스로인한정신상담률_표준화율']
    x_lst2 = ['알루미늄(기준:0.2/ 단위:(mg/L))', '과망간산칼륨소비량(기준:10/ 단위:(mg/L))', '염소이온(기준:250/ 단위:(mg/L))', '불소(기준:1.5/ 단위:(mg/L))', '디브로모클로로메탄(기준:0.1/ 단위:(mg/L))', '질산성질소(기준:10/ 단위:(mg/L))', '수소이온농도(기준:5.8 ~ 8.5/ 단위:-)', '1,2-디브로모-3-클로로프로판(기준:0.003/ 단위:(mg/L))', '톨루엔(기준:0.7/ 단위:(mg/L))']
    y_lst2 = ['우울감경험률_표준화율']
    x_lst3 = ['염소이온(기준:250/ 단위:(mg/L))', '불소(기준:1.5/ 단위:(mg/L))', '붕소(기준:1/ 단위:(mg/L))', '잔류염소(기준:4/ 단위:(mg/L))', '증발잔류물(기준:500/ 단위:(mg/L))', '클로로포름(기준:0.08/ 단위:(mg/L))', '사염화탄소(기준:0.002/ 단위:(mg/L))']
    y_lst3 = ['주관적구강건강이나쁜인구의분율_표준화율']
    x_lst4 = ['염소이온(기준:250/ 단위:(mg/L))', '불소(기준:1.5/ 단위:(mg/L))', '증발잔류물(기준:500/ 단위:(mg/L))', '클로로포름(기준:0.08/ 단위:(mg/L))', '황산이온(기준:200/ 단위:(mg/L))', '세제(기준:0.5/ 단위:(mg/L))', '브로모디클로로메탄(기준:0.03/ 단위:(mg/L))', '클로랄하이드레이트(기준:0.03/ 단위:(mg/L))', '크롬(기준:0.05/ 단위:(mg/L))']
    y_lst4 = ['스트레스로인한정신상담률_표준화율']
    x_lst5 = ['불소(기준:1.5/ 단위:(mg/L))', '증발잔류물(기준:500/ 단위:(mg/L))', '황산이온(기준:200/ 단위:(mg/L))', '알루미늄(기준:0.2/ 단위:(mg/L))', '망간(기준:0.05/ 단위:(mg/L))']
    y_lst5 = ['우울증상으로인한정신상담률_표준화율']

    # Multi Linear Regression 실행
    water = MLR.waterPreprocessing(x_lst1)
    x1 = MLR.waterQualMean()
    y1 = MLR.healthPreprocessing(y_lst1)
    score = MLR.mlr(x1,y1)
    print(score)
