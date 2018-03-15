import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

#Create Dataset with data for total population
def create_df(df, prefix, group_col,new_name,columns_perc):
    columns = ['Id','County','State']
    cols = [prefix + '_' + S for S in group_col]
    columns.extend(cols)
    df = df[columns]
    rename = {}
    for k,v in new_name.items():
        rename [prefix + '_' + k] = v
        df[prefix + '_' + k] = df[prefix + '_' + k].astype('float64')
    for c in columns_perc:
        df[prefix + '_' + c] = df[prefix + '_' + 'VC03'] * df[prefix + '_' + c]/100
    df['Id'] = df['Id'].astype('float64')
    df.rename(columns=rename, inplace=True)
    return df

def data_select():
    df2016 = pd.read_csv('data/ACS_16_5YR_CP05_with_ann.csv',encoding='iso-8859-1')
    title = df2016[df2016['GEO.id'] == 'Id']
    df2016 = df2016[df2016['GEO.id'] != 'Id']
    df2016.rename(columns={'GEO.id2':'Id'}, inplace=True)

    df2016['County'] = df2016['GEO.display-label'].apply(lambda x: (re.split(" County, ",str(x))[0]))
    df2016['State'] = df2016['GEO.display-label'].apply(lambda x: (re.split(", ",str(x))[1]))

    prefix_11_16 = 'HC01'    # Date between 2011 and 2016
    #prefix_06_10 = 'HC02'    # Date between 2006 and 2011
    group_col = ['VC03','VC04','VC05','VC08','VC09','VC10',
                 #'VC11','VC12','VC13','VC14','VC15','VC16',
                 'VC17','VC18','VC19','VC20']

    columns_perc = ['VC04','VC05','VC08','VC09','VC10',
                    #'VC11','VC12','VC13','VC14','VC15','VC16',
                    'VC17','VC18','VC19','VC20']

    new_name  = {'VC03':'Total_pop',
                 'VC04':'Total_man',
                 'VC05':'Total_wom',
                 'VC08':'Age_00_04y',
                 'VC09':'Age_05_09y',
                 'VC10':'Age_10_14y',
    #             'VC11':'Age_15_19y',
    #             'VC12':'Age_20_24y',
    #             'VC13':'Age_25_34y',
    #             'VC14':'Age_35_44y',
    #             'VC15':'Age_45_54y',
    #             'VC16':'Age_55_59y',
                 'VC17':'Age_60_64y',
                 'VC18':'Age_65_74y',
                 'VC19':'Age_75_84y',
                 'VC20':'Age_85_mor'}

    df2016_pop = create_df(df2016, prefix_11_16,group_col,new_name,columns_perc)

    #Data Set - Industry by County
    temp = pd.read_csv('data/ECN_2012_US_31A1_with_ann.csv',encoding='iso-8859-1')
    columns = ['GEO.id2','ESTAB', 'ECTGE20','EMP','PAYANN','EMPAVPW','HOURS','PAYANPW','RCPTOT']
    title_industry = temp[temp['GEO.id'] == 'Id']
    temp = temp[temp['GEO.id'] != 'Id']
    temp = temp[temp['NAICS.display-label']=='Manufacturing']
    temp = temp[columns]
    for col in columns:
        temp[col] = pd.to_numeric(temp[col],errors = 'coerce')
    temp = temp.fillna(0)
    temp.rename(columns={'GEO.id2':'Id','ESTAB':'num_estab','ECTGE20':'more_20employ','EMP':'num_employees','PAYANN':'total_pay','EMPAVPW':'prod_wokers','HOURS':'prod_hours','PAYANPW':'pay_workes','RCPTOT':'value_ship'}, inplace=True)
    df2016_industry = temp

    # Income Percapta per County
    temp = pd.read_csv('data/ACS_16_5YR_DP03_with_ann.csv',encoding='iso-8859-1')
    columns = ['GEO.id2','HC01_VC118','HC01_VC66','HC01_VC130']
    title = temp[temp['GEO.id'] == 'Id']
    temp = temp[temp['GEO.id'] != 'Id']
    df2016_income = temp[columns]
    for col in columns:
        df2016_income[col] = df2016_income[col].astype('float64')
    df2016_income.rename(columns={'GEO.id2':'Id','HC01_VC118':'Income_percapta','HC01_VC66':'Worker','HC01_VC130':'Insurance',}, inplace=True)

    df2016_main = pd.merge(pd.merge (df2016_pop, df2016_income, on=['Id']),df2016_industry, on=['Id'])
    df2016_main['Size_County'] = np.where(df2016_main['Total_pop'] < 50000,'County under 50k people','County over 50k people')
    df2016_main['perc_man']= ((df2016_main['Total_man']/df2016_main['Total_pop'])*100)
    df2016_main['perc_kids']= (((df2016_main['Age_00_04y']+df2016_main['Age_05_09y']+df2016_main['Age_10_14y'])/df2016_main['Total_pop'])*100)
    df2016_main['perc_seniors']= (((df2016_main['Age_60_64y']+df2016_main['Age_60_64y']+df2016_main['Age_65_74y']+df2016_main['Age_75_84y']+df2016_main['Age_85_mor'])/df2016_main['Total_pop'])*100)
    df2016_main['perc_worker']= ((df2016_main['Worker']/df2016_main['Total_pop'])*100)
    df2016_main['perc_no_insurance']= ((df2016_main['Insurance']/df2016_main['Total_pop'])*100)

    df2016 = df2016_main[['Id', 'County', 'State', 'Size_County', 'Total_pop',
                          'Income_percapta', 'num_estab',
                          'more_20employ', 'num_employees',
                          'total_pay', 'prod_wokers', 'prod_hours',
                          'pay_workes', 'value_ship','perc_man', 'perc_kids', 'perc_worker',
                          'perc_no_insurance','perc_seniors']]

    return df2016

if __name__=='__main__':
    a = data_select()
    print(a.head())
