# Добавление 2ух колонок: 
#    people_count - количество людей в отделе
#    sick_count - количесвто заболевших людей в отеделе
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