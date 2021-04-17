import pandas as pd
from transliterate import translit


def add_rolling_mean(data, column, period):
    data[f"{column}_rolling_mean_{period}"] = data.sort_values(["hash_tab_num", "date"]).groupby('hash_tab_num').rolling(period, min_periods=0).agg({column: "mean"}).reset_index(drop=True)
    return data


def add_cummean(data, column):
    data = data.copy(deep=True)
    data[['cumsum', "cumcount", f"{column}_cummax"]] = data.sort_values(["hash_tab_num", "date"]).groupby('hash_tab_num').agg({column: ["cumsum", "cumcount", "cummax"]})
    data["cumcount"] = data["cumcount"] + 1
    data[f"{column}_cummean"] = data['cumsum'] / data["cumcount"]

    data = add_rolling_mean(data, column, 2)
    data = add_rolling_mean(data, column, 3)
    data = add_rolling_mean(data, column, 6)
    data = add_rolling_mean(data, column, 9)
    data = add_rolling_mean(data, column, 12)
    data = add_rolling_mean(data, column, 24)

    data[f"trend_{column}_2_24"] = data[f"{column}_rolling_mean_2"] / data[f"{column}_rolling_mean_24"]
    data[f"trend_{column}_2_12"] = data[f"{column}_rolling_mean_2"] / data[f"{column}_rolling_mean_12"]

    return data.drop(columns=['cumsum', "cumcount"])

def generate_features(sot, rod, ogrv, weather):

    # Создание вспомогательного датафрейма с информацией о количестве смен сотрудника в месяце
    ogrv['month'] = ogrv['date'].map(lambda x: x[0:8] + str('01'))
    kolvo_smen = ogrv[ogrv.work_shift_type.isin(['Смена 1', 'Смена 2', 'Смена 3'])]\
    [['hash_tab_num','month','work_shift_type']].groupby(['hash_tab_num','month']).agg('count').reset_index()
    kolvo_smen.columns = ['hash_tab_num', 'date', 'work_shift_type_count'] 

    # Создание вспомогательного датафрейма с информацией о сумме рабочих часов в месяце
    ogrv['number_of_working_hours'] = ogrv['number_of_working_hours'].astype('str')
    ogrv['number_of_working_hours'] = ogrv['number_of_working_hours'].apply(lambda x: float(x.replace(',','.')))
    sum_work_hours = ogrv[ogrv.work_shift_type.isin(['Смена 1', 'Смена 2', 'Смена 3'])]\
    [['hash_tab_num','month','number_of_working_hours']].groupby(['hash_tab_num','month']).agg('sum').reset_index()
    sum_work_hours.columns = ['hash_tab_num', 'date', 'sum_work_hours']


    # Создание вспомогательного датафрейма с информацией о факте больничного в текущем месяце
    kolvo_bolni4 = ogrv[ogrv.graphic_rule_level_1.isin(['Больничный'])]\
    [['hash_tab_num','month','graphic_rule_level_1']].groupby(['hash_tab_num','month']).agg('count').reset_index()
    kolvo_bolni4['graphic_rule_level_1'] = 1
    kolvo_bolni4.columns = ['hash_tab_num', 'date', 'sick']

    
    # Количество дней в месяце в категориях Больничные, Выходной, Прогулы и т.д.
    ogrv = pd.get_dummies(ogrv, columns = ['graphic_rule_level_1'])
    ogrv.columns  = [translit(column,'ru', reversed=True).replace("'","") for column in ogrv.columns]
    list_column_type_of_day = [column for column in ogrv.columns if 'graphic_rule_level_1' in column]
    result_cnt_category_days = ogrv[['hash_tab_num','month']].rename({'month': 'date'}, axis=1)
    for column in list_column_type_of_day:
        cnt_category_days= ogrv[['hash_tab_num','month', column]].groupby(['hash_tab_num','month']).agg('sum').reset_index()
        cnt_category_days.columns = ['hash_tab_num', 'date', 'cnt_days_'+column]
        result_cnt_category_days = pd.merge(result_cnt_category_days, cnt_category_days, how = 'left', on = ['hash_tab_num','date'])

    #    name_post_lvl4_people_count - количество людей в отделе
    #    name_post_lvl4_sick_count - количесвто заболевших людей в отеделе
    #    name_post_lvl4_sick_avg - доля заболевших людей в отеделе
    df_name_post_lvl4_agg =\
            sot \
                .merge(
                    sot \
                        .groupby(['date', 'name_post_lvl4']) \
                        .agg(
                            name_post_lvl4_people_count=("hash_tab_num", "count"),
                            name_post_lvl4_sick_count=("sick", "sum"),
                            name_post_lvl4_sick_avg=("sick", "mean")
                        ) \
                        .reset_index(),
                    on=['date', 'name_post_lvl4'],
                    how='left'
                )[['hash_tab_num', 'date','name_post_lvl4_people_count','name_post_lvl4_sick_count','name_post_lvl4_sick_avg']]

    df_name_fact_lvl4_agg =\
            sot \
                .merge(
                    sot \
                        .groupby(['date', 'name_fact_lvl4']) \
                        .agg(
                            name_fact_lvl4_people_count=("hash_tab_num", "count"),
                            name_fact_lvl4_sick_count=("sick", "sum"),
                            name_fact_lvl4_sick_avg=("sick", "mean")
                        ) \
                        .reset_index(),
                    on=['date', 'name_fact_lvl4'],
                    how='left'
                )[['hash_tab_num', 'date','name_fact_lvl4_people_count','name_fact_lvl4_sick_count','name_fact_lvl4_sick_avg']]

    # Возраст
    sot['age'] = ([int(x[0:4]) for x in sot['date']] - sot['date_of_birth'])

    # Базовый датафремй
    sot_data = sot[['hash_tab_num','date','category', 'age', 'is_local','gender','razryad_fact', 'razryad_post', 'work_experience_company',
                    'name_fact_lvl5','education','home_to_work_distance']]
    sot_data.gender = sot_data['gender'].map(lambda x: 1 if x == 'мужской' else 0)

    # Создание вспомогательно датасета с информацией о родственниках - пенсионерах
    # (55 лет для женщин и 60 лет для мужчин для региона севера)
    sot_data = pd.merge(sot_data,rod, how = 'left', on = 'hash_tab_num')
    sot_data['rel_cur_old'] = ([int(x[0:4]) for x in sot_data['date']] - sot_data['rel_birth'])
    sot_data['rel_is_male'] = sot_data.rel_type.map(lambda x:1 if x \
        in ['Сын', 'Муж', 'Отец', 'Пасынок', 'Внук','Брат'] else 0)

    retiree = sot_data[((sot_data.rel_cur_old > 55) & (sot_data.rel_is_male == 0) \
                | (sot_data.rel_cur_old > 60) & (sot_data.rel_is_male == 1))]\
        [['hash_tab_num','date','rel_is_male']].groupby(['hash_tab_num','date']).agg('count').reset_index()
    retiree.columns = ['hash_tab_num','date','rale_is_old']

    # Добавим фичи о количестве малолетних детей
    sot_data['rel_is_children'] = sot_data.rel_type.map(lambda x:1 if x \
        in ['Сын', 'Дочь', 'Пасынок', 'Падчерица', 'Опекаемый (воспитанник)','Опекаемая (воспитанница)'] else 0)
    young_children_6_cnt = sot_data[(sot_data.rel_is_children==1)&(sot_data.rel_cur_old<=6)][['hash_tab_num','date','rel_is_children']].groupby(['hash_tab_num','date']).agg('count').reset_index()
    young_children_6_cnt.columns = ['hash_tab_num','date','young_children_6_cnt']
    young_children_11_cnt = sot_data[(sot_data.rel_is_children==1)&(sot_data.rel_cur_old<=12)][['hash_tab_num','date','rel_is_children']].groupby(['hash_tab_num','date']).agg('count').reset_index()
    young_children_11_cnt.columns = ['hash_tab_num','date','young_children_11_cnt']
    young_children_6_female_cnt = sot_data[(sot_data.rel_is_children==1)&(sot_data.rel_cur_old<=6)&(sot_data.gender==0)][['hash_tab_num','date','rel_is_children']].groupby(['hash_tab_num','date']).agg('count').reset_index()
    young_children_6_female_cnt.columns = ['hash_tab_num','date','young_children_6_female_cnt']
    young_children_11_female_cnt = sot_data[(sot_data.rel_is_children==1)&(sot_data.rel_cur_old<=12)&(sot_data.gender==0)][['hash_tab_num','date','rel_is_children']].groupby(['hash_tab_num','date']).agg('count').reset_index()
    young_children_11_female_cnt.columns = ['hash_tab_num','date','young_children_11_female_cnt']
    sot_data.drop(['rel_type','rel_birth','rel_cur_old','rel_is_male','rel_is_children'], axis = 1, inplace = True)
    # Создание вспомогательно датасета с информацией о количестве сотрудников в подразделении
    # по фактическому месту работы
    division_count = sot_data[['hash_tab_num','date','name_fact_lvl5']].\
    groupby(['name_fact_lvl5','date']).agg('count').reset_index()
    division_count.columns = ['name_fact_lvl5', 'date', 'personel_num']
    sot_data = pd.merge(sot_data, division_count, how = 'left', on = ['date','name_fact_lvl5'])

    # Создание dummy переменных
    sot_data.education = sot_data['education']\
    .map(lambda x: 'Высшее' if x in ['Высшее образование','Высшее-бакалавриат','Высшее-специалитет'] else(\
    'Среднее_профессинальное' if x in ['Ср.профессиональное','Нач.профессиональное'] else 'Начальное_среднее'))
    sot_data = pd.get_dummies(sot_data, columns = ['category','education','razryad_fact','razryad_post'])\
    .drop('name_fact_lvl5', axis = 1)

    # Создание единого датасета для будущего использования в модели
    merged_data = pd.merge(sot_data, retiree, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, young_children_6_cnt, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, young_children_11_cnt, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, young_children_6_female_cnt, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, young_children_11_female_cnt, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, kolvo_smen, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, sum_work_hours, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, kolvo_bolni4, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, result_cnt_category_days, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, df_name_post_lvl4_agg, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, df_name_fact_lvl4_agg, how = 'left', on = ['hash_tab_num','date'])
    merged_data = merged_data.drop_duplicates()



    # Создание 12ти столбцов с датами будущих периодов для формирования таргетов
    merged_data['sick'] = merged_data['sick'].fillna(0)
    merged_data['target_dates'] = merged_data['date'].apply(lambda x: pd.date_range((x),\
        periods = 13, freq='1MS',closed = 'right'))
    new_target_dates = pd.DataFrame(merged_data['target_dates'].tolist(), \
        columns = ['y_dt_'+str(i) for i in range(1,13)], index = merged_data.index)
    merged_data = pd.merge(merged_data,new_target_dates, left_index=True, right_index=True)
    merged_data.drop(['target_dates'],axis = 1, inplace = True)
    merged_data['date'] = pd.to_datetime(merged_data['date'])


    # Добавим информацию о погоде
    merged_data["month"] = merged_data["date"].dt.month
    merged_data = merged_data.merge(weather, left_on='month', right_on="Месяц").drop(columns=["Месяц"])
    merged_data.columns  = [translit(column,'ru', reversed=True).replace("'","").replace(" ",'_') for column in merged_data.columns]

    # Присоединение данных о больничных к будущим периодам созданным на предыдущем шаге
    for i in range(1,13):
        dt_col_name = 'y_dt_'+str(i)
        y_col_name = 'y_'+str(i)
        targets_tmp = merged_data[['date','hash_tab_num','sick']]
        targets_tmp.columns = [dt_col_name, 'hash_tab_num', y_col_name]
        merged_data = pd.merge(merged_data, targets_tmp, how = 'left', on = [dt_col_name, 'hash_tab_num'])
        merged_data.drop(dt_col_name, axis = 1, inplace = True)


    
    y = merged_data[['date', 'hash_tab_num', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12']]
    y_col_names= ['y_' + str(i)  for i in range(1,13)]
    X = merged_data.drop(['y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12'], axis = 1)
    



    return X, y
