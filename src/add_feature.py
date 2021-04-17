# Добавление 2ух колонок: 
#    people_count - количество людей в отделе
#    sick_count - количество заболевших людей в отделе
sot \
    .merge(
        sot \
            .groupby(['date', 'name_post_lvl4']) \
            .agg(
                people_count=("hash_tab_num", "count"),
                sick_count=("sick", "sum")
            ) \
            .reset_index(),
        on=['date', 'name_post_lvl4'],
        how='left'
    )

# Добавление колонки health_streak 
# Сколько месяцев подряд не болел сотрудник
def health_streak(x):
    x = x.values
    zero_count = 0
    res = []
    for i in range(len(x)):
        if (x[i] == 0):
            res.append(zero_count)
            zero_count += 1
        else:
            zero_count = 0
            res.append(zero_count)
    return res

sot_ord = sot.sort_values(by=['hash_tab_num', 'date'])

health = sot_ord \
    .groupby('hash_tab_num') \
    .agg(
        health_streak=("sick", lambda x: health_streak(x))
    ) \
    .health_streak \
    .to_numpy()

sot_ord['health_streak'] = [item for sublist in health for item in sublist]