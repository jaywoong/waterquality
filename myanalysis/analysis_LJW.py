import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from config.settings import DATA_DIRS


# ------------------------------------------------------------------------------------------------------

class WnH:
    def healthPreprocessing(self):
        # 2009 ~ 2018년 건강 데이터
        dfh_all = pd.read_excel(DATA_DIRS[0] + '//health_2009_2018_preprocessed.xlsx', sheet_name = None, engine = 'openpyxl');
        dfh = pd.concat(dfh_all, ignore_index=True);

        # 분석 대상 지역 데이터 추출
        area = dfh['지역'].isin(['서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시','울산광역시','경기도','강원도','충청북도','충청남도','전라북도','전라남도','경상북도','경상남도','제주특별자치도'])
        dfh = dfh[area]

        # 분석 대상 건강지표 컬럼 추출
        health = dfh[['시도', '스트레스인지율_표준화율', '우울감경험률_표준화율', '주관적구강건강이나쁜인구의분율_표준화율', '스트레스로인한정신상담률_표준화율']]
        health = health.set_index('시도')

        # 건강 데이터 전처리 데이터프레임: health
        return health

    def waterPreprocessing(self):
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
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        # Test
        pd.DataFrame(lr.predict(x_test))
        score = lr.score(x_test, y_test)

        return x_scaled, y, score

    def varSelection(self, x_scaled, y):         #변수선택법(단계별 선택법)

        colName = water.columns[3:]
        df = pd.DataFrame(x_scaled, columns=colName)
        df_y = y

        ## 전진 단계별 선택법
        variables = df.columns.tolist() ## 설명 변수 리스트

        ## 반응 변수 = 'y의 컬럼들'
        selected_variables = [] ## 선택된 변수들
        sl_enter = 0.05
        sl_remove = 0.05

        sv_per_step = [] ## 각 스텝별로 선택된 변수들
        adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
        steps = [] ## 스텝
        step = 0
        for i in range(0,6):
            y = df_y.iloc[:,i]
            y = list(y)
            while len(variables) > 0:
                remainder = list(set(variables) - set(selected_variables))
                pval = pd.Series(index=remainder) ## 변수의 p-value
                ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서
                ## 선형 모형을 적합한다.
                for col in remainder:
                    X = df[selected_variables+[col]]
                    X = sm.add_constant(X)
                    model = sm.OLS(y,X).fit()
                    pval[col] = model.pvalues[col]

                min_pval = pval.min()
                if min_pval < sl_enter: ## 최소 p-value 값이 기준 값보다 작으면 포함
                    selected_variables.append(pval.idxmin())
                    ## 선택된 변수들에대해서
                    ## 어떤 변수를 제거할지 고른다.
                    while len(selected_variables) > 0:
                        selected_X = df[selected_variables]
                        selected_X = sm.add_constant(selected_X)
                        selected_pval = sm.OLS(y,selected_X).fit().pvalues[1:] ## 절편항의 p-value는 뺀다
                        max_pval = selected_pval.max()
                        if max_pval >= sl_remove: ## 최대 p-value값이 기준값보다 크거나 같으면 제외
                            remove_variable = selected_pval.idxmax()
                            selected_variables.remove(remove_variable)
                        else:
                            break

                    step += 1
                    steps.append(step)
                    adj_r_squared = sm.OLS(y,sm.add_constant(df[selected_variables])).fit().rsquared_adj
                    adjusted_r_squared.append(adj_r_squared)
                    sv_per_step.append(selected_variables.copy())
                else:
                    break
            print(selected_variables)
        return
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    WnH = WnH()

    water = WnH.waterPreprocessing()
    x = WnH.waterQualMean()
    y = WnH.healthPreprocessing()

    x_scaled, y, score = WnH.mlr(x,y)
    print(score)
    WnH.varSelection(x_scaled,y)
