import pandas as pd

def generate_features(sot, rod, ogrv):

    # Создание вспомогательного датафрейма с информацией о количестве смен сотрудника в месяце
    ogrv['month'] = ogrv['date'].map(lambda x: x[0:8] + str('01'))
    kolvo_smen = ogrv[ogrv.work_shift_type.isin(['Смена 1', 'Смена 2', 'Смена 3'])]\
    [['hash_tab_num','month','work_shift_type']].groupby(['hash_tab_num','month']).agg('count').reset_index()
    kolvo_smen.columns = ['hash_tab_num', 'date', 'work_shift_type_count'] 

    # Создание вспомогательного датафрейма с информацией о факте больничного в текущем месяце
    kolvo_bolni4 = ogrv[ogrv.graphic_rule_level_1.isin(['Больничный'])]\
    [['hash_tab_num','month','graphic_rule_level_1']].groupby(['hash_tab_num','month']).agg('count').reset_index()
    kolvo_bolni4['graphic_rule_level_1'] = 1
    kolvo_bolni4.columns = ['hash_tab_num', 'date', 'sick']

    # Базовый датафремй
    sot_data = sot[['hash_tab_num','date','category','gender','razryad_fact','work_experience_company',
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
    sot_data.drop(['rel_type','rel_birth','rel_cur_old','rel_is_male'], axis = 1, inplace = True)
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
    sot_data = pd.get_dummies(sot_data, columns = ['category','education','razryad_fact'])\
    .drop('name_fact_lvl5', axis = 1)

    # Создание единого датасета для будущего использования в модели
    merged_data = pd.merge(sot_data, retiree, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, kolvo_smen, how = 'left', on = ['hash_tab_num','date'])
    merged_data = pd.merge(merged_data, kolvo_bolni4, how = 'left', on = ['hash_tab_num','date'])
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
    # Присоединение данных о больничных к будущим периодам созданным на предыдущем шаге
    for i in range(1,13):
        dt_col_name = 'y_dt_'+str(i)
        y_col_name = 'y_'+str(i)
        targets_tmp = merged_data[['date','hash_tab_num','sick']]
        targets_tmp.columns = [dt_col_name, 'hash_tab_num', y_col_name]
        merged_data = pd.merge(merged_data, targets_tmp, how = 'left', on = [dt_col_name, 'hash_tab_num'])
        merged_data.drop(dt_col_name, axis = 1, inplace = True)

    return merged_data