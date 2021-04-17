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

## Добавлена колонка ялвяется ли последним днем месяца при больничном
from calendar import monthrange

def generate_period(ogrv):
    
    def is_last_day(date):
        return monthrange(date.year, date.month)[1] == date.day

    ogrv_b = ogrv \
        .query("graphic_rule_level_2 == 'Б'") \
        .sort_values(by=['hash_tab_num', 'date']) 
    ogrv_b['date'] = pd.to_datetime(ogrv_b['date'])
    ogrv_b['is_last_day'] = ogrv_b.date.apply(lambda x: is_last_day(x))
    ogrv_b['date'] = ogrv_b.date.apply(lambda x: x.replace(day=1))
    ogrv_b = ogrv_b \
        .groupby(['hash_tab_num', 'date']) \
        .agg(
            is_last_day=('is_last_day', 'sum')
        ) \
        .reset_index()
    ogrv_b['is_last_day'] = ogrv_b.is_last_day.astype('int')
    return ogrv_b
       
#res = generate_period(ogrv)