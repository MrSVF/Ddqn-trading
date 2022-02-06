import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment

from em import equity

def convertDataToTimeStep(data, step_type):
    st = tf.constant(np.array(np.array([step_type], dtype=np.int32)))
    rw = tf.constant(np.array(np.array([0], dtype=np.float32)))  
    
    disc = 0 
    if(step_type == 0):
        disc = 1

    ds = tf.constant(np.array(np.array([disc], dtype=np.float32)))
    ts_obs = tf.convert_to_tensor(np.array(np.array([data], dtype=np.float64)))

    t = ts.TimeStep(st, rw, ds, ts_obs)
    return t

def runPolicy(t, _policy, _policy_state):
    if(t.step_type.numpy()[0] == 0):
        _policy_state = _policy.get_initial_state(1)
        
    a, _policy_state, _ = _policy.action(t, _policy_state)
    return a, _policy_state


# Настройки
#---------------------------------------------------------------------------------------
policy_dir = './chkp/(150, 13)_policy_max/' # Путь к сохраненной политике
window_size = 150                    # Ширина окна данных
data_begin  = 0000                   # Начальный индекс истории
data_end    = 507200                 # Конечный индекс истории 
#---------------------------------------------------------------------------------------


# Чтение файла истории
#---------------------------------------------------------------------------------------
df = pd.read_csv(r'data_20-21.txt', sep=';')[data_begin : data_end]
dfa = df[[s for s in df.columns if 'MSE' in s]]
dfb = dfa[[s for s in dfa.columns]] / (dfa[[s for s in dfa.columns]].shift())
# dfc = pd.concat([dfa, dfb], axis=1)
dfx = np.log(dfb)
dfx = dfx.dropna()
dfx_ind = dfx.index
market_data = dfx.values
real_price = df['REAL_PRICE'][dfx_ind].values


# Чтение политики
#---------------------------------------------------------------------------------------
policy = tf.compat.v2.saved_model.load(policy_dir)
#---------------------------------------------------------------------------------------


# Тестер
#---------------------------------------------------------------------------------------
lendf = len(market_data)         # Длина датафрейма

# Константы позиции
LONG  = 0
SHORT = 1
FLAT  = 2

# Константы действий
BUY  = 0
SELL = 1
HOLD = 2

windowPositions = np.zeros((window_size, 3)) # История позиций, добавляется к наблюденям
windowPrifit = np.zeros((window_size, 1))    # История профитов, добавляется к наблюденям (сейчас не реализовано)
windowPosDuration = np.zeros((window_size, 1))

one_hot_position = np.eye(3)[FLAT]           # Кодировка позиции в формате one_hot_encoding
reward = 0          # Награда
portfolio = 0       # Сумма наград
n_short   = 0       # Количество шортов
n_long    = 0       # Количество лонгов
position  = FLAT    # Начальная позиция
e1 = equity()       # Эквити из emodule
eq1 = []            # Список эквити
eq2 = []            # Список эквити2
PosDuration = 0

s = market_data[0 : window_size] # Начальное наблюдение, только маркет дата

windowPositions = np.roll(windowPositions, -1, axis=0)  # Шаг по окну позиций
windowPositions[s.shape[0]-1] = one_hot_position        # Добавление текущей позиции
windowPrifit = np.roll(windowPrifit, -1, axis=0)        # Шаг по окну профитов
windowPrifit[s.shape[0]-1] = reward                     # Добавление текущего профита
windowPosDuration = np.roll(windowPosDuration, -1, axis=0)
windowPosDuration[s.shape[0]-1] = PosDuration

# observation = np.concatenate((np.delete(s, 0, 1), windowPositions, windowPosDuration), axis=1)  # Формирование начального наблюдения
observation = np.concatenate((s, windowPositions), axis=1)
t = convertDataToTimeStep(observation, 0)                                                       # Формирование начального time step

policy_state = policy.get_initial_state(1)              # Начальное состояние политики
act, policy_state = runPolicy(t, policy, policy_state)  # Получение действия от агента (политики)

# Цикл тестера
# ---------------------------------------------------------------------------------------
for i in range(1, lendf-window_size + 1):   # Начало со второй строки данных, по первой получаем первое действие
    
    s = market_data[i : i + window_size]               # Текущее наблюдение, только маркет дата
    reward = 0                              # Награда
    closingPrice = real_price[i + window_size - 1]       # Цена закрытия, берется из последнего элемента первого столбца массива наблюдений

# Торговая логика
#----------------------------------------------------------------------------------------
    if act == BUY:      # Действие агента покупка
        action = BUY    # Состояние покупка

        if position == FLAT:    # Если предыдущая позиция флет
            position = LONG     # Меняем позицию на лонг
            entry_price = closingPrice                  # Запоминаем цену входа в лонг
            e1.add(trd_price=closingPrice, trd_qty=1)   # Добавляем покупку в emodule
            # eq1.append(e1.e_val)                        # Запоминаем объём в emodule

        elif position == SHORT:     # Если предыдущая позиция шорт
            position = FLAT         # Меняем позицию на флет
            exit_price = closingPrice                   # Запоминаем цену покупки
            # reward = entry_price - exit_price
            # rewards.append(reward)
            e1.add(trd_price=closingPrice, trd_qty= -e1.qty )    # # Закрываем позицию  в emodule
            # eq1.append(e1.e_val)                                # Запоминаем объём в emodule

            entry_price = 0         # Сбрасываем цену входа
            n_short += 1            # Увеличиваем счетик шортов

    elif act == SELL:   # Действие агента продажа
        action = SELL   # Состояние продажа

        if position == FLAT:        # Если предыдущая позиция флет
            position = SHORT        # Меняем позицию на шорт
            entry_price = closingPrice                  # Запоминаем цену входа в шорт
            e1.add(trd_price=closingPrice, trd_qty=-1)  # Добавляем продажу в emodule
            # eq1.append(e1.e_val)                        # Запоминаем объём в emodule

        elif position == LONG:      # Если предыдущая позиция лонг
            position = FLAT
            exit_price = closingPrice                   # Запоминаем цену продажи
            # reward = exit_price - entry_price
            # rewards.append(reward)
            e1.add(trd_price=closingPrice, trd_qty= -e1.qty )    # Закрываем позицию в emodule
            # eq1.append(e1.e_val)                                # Запоминаем объём в emodule

            entry_price = 0         # Сбрасываем цену входа
            n_long += 1             # Увеличиваем счетик лонгов

    # if e1.e_val != 0:
    eq1.append(e1.e_val)            # Запоминаем объём в emodule
    eq2.append( e1.e_val+(closingPrice - e1.price)*e1.qty )

    if (position != FLAT):
        PosDuration += 1/400
    else:
        PosDuration = 0
#----------------------------------------------------------------------------------------

    one_hot_position = np.eye(3)[position]                  # Кодировка позиции в формате one_hot_encoding

    windowPositions = np.roll(windowPositions, -1, axis=0)  # Шаг по окну позиций
    windowPositions[s.shape[0]-1] = one_hot_position        # Добавление текущей позиции
    windowPrifit = np.roll(windowPrifit, -1, axis=0)        # Шаг по окну профитов
    windowPrifit[s.shape[0]-1] = reward                     # Добавление текущего профита
    windowPosDuration = np.roll(windowPosDuration, -1, axis=0)
    windowPosDuration[s.shape[0]-1] = PosDuration

    # observation = np.concatenate((np.delete(s, 0, 1), windowPositions, windowPosDuration), axis=1) # Формирование текущего наблюдения
    observation = np.concatenate((s, windowPositions), axis=1)

    if (i >= lendf - window_size):                          # Условие последнего шага
        t = convertDataToTimeStep(observation, 2)           # Формирование последнего time step
    else:
        t = convertDataToTimeStep(observation, 1)           # Формирование текущего time step

    act, policy_state = runPolicy(t, policy, policy_state)  # Получение действия от агента (политики)
    
    portfolio += reward                                     # Добавление награды в портфель


max_dd = 1
eq1[0] = 1
if len(eq1) > 3:
    rolling_max = np.maximum.accumulate(eq1)
    max_dd = np.max((rolling_max - eq1)/rolling_max)

print('\nЛонгов: ', n_long)
print('Шортов: ', n_short)
plt.plot(eq1)   # График эквити
# plt.plot(eq2)   # График эквити
plt.xlabel('Сделок: {0}, L: {1}, S: {2}, Max_dd: {3:.2f}'.format(n_long+n_short, n_long, n_short, max_dd))
plt.show()      # Показать окно графика