# NorNickel_hackathon_2021
# Задача
https://www.notion.so/2-a46d65ff68d54af0af162a65d2fdabd5
# Подход
* Прогноз строится при помощи 12 моделей, каждая из которых прогнозирует на определенной количество месяцев вперед: Первая модель прогнозирует возьмет ли сотрудниик больничный в следующем месяце, вторая прогнозирует возьмет ли сотрудник больничный через месяц и. т. д.
* Трэшхолды подбираются так, что бы максимизироват F1 для последних трех точек
* Гиперпараметры подбирались через hyperopt

# Осноновные генерируемые фичи
 * Факт болезни в последний день предыдущего месяца
 * Наличиее малееньких детей и женский пол
 * Стаж работы, в том числе разбитый на бины
 * Различный статистики по количеству больничных в предыдущие месяца
 * Статистика по количееству смен
 ...

# Что попробовали, но не взлетело
* Построение единой модели для всех прогнозных месяцов
* Использование верхнеуровнего прогноза количества заболевших всего, полученного от prophet
* Использование усредненных данных о погоде в Мурманске
