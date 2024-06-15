import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import DistanceMetric
from joblib import dump, load
# Признаки
numeric_features = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']


st.title("Предсказание свойств материала")

# Создаем виджеты для ввода признаков в правой колонке
with st.sidebar:
    st.title("Входные параметры")
    f_1 = st.number_input('Соотношение матрица-наполнитель',format = '%.6f', value=3.0000, step=0.0001)
    f_2 = st.number_input('Плотность, кг/м3', format = '%.6f', value=2000.0, step=0.0001)
    f_3 = st.number_input('модуль упругости, ГПа', format = '%.6f', value=2000.0, step=0.0001)
    f_4 = st.number_input('Количество отвердителя, м.%', format = '%.6f', value=2000.0, step=0.0001)
    f_5 = st.number_input('Содержание эпоксидных групп,%_2', format = '%.6f', value=2000.0, step=0.0001)
    f_6 = st.number_input('Температура вспышки, С_2', format = '%.6f', value=2000.0, step=0.0001)
    f_7 = st.number_input('Поверхностная плотность, г/м2', format = '%.6f', value=2000.0, step=0.0001)
    f_8 = st.number_input('Потребление смолы, г/м2', format = '%.6f', value=2000.0, step=0.0001)
    f_9 = st.number_input('Угол нашивки, град', format = '%.6f', value=2000.0, step=0.0001)
    f_10 = st.number_input('Шаг нашивки', format = '%.6f', value=2000.0, step=0.0001)
    f_11 = st.number_input('Плотность нашивки', format = '%.6f', value=2000.0, step=0.0001)

model_final_pr = load('model_1.joblib')

features = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11]
df_features = pd.DataFrame([features], columns=model_final_pr.feature_names_in_)
predict_1 = model_final_pr.predict(df_features)[0]

# st.text(f'Предсказание для прочности при растяжении: {predict_1} МПа')

# Вывод предсказания с жирным шрифтом для значения
st.markdown(
    f"""
    <div>
        Предсказание для прочности при растяжении: <strong>{predict_1:.3f}</strong> МПа
    </div>
    """,
    unsafe_allow_html=True
)

model_final_upr = load('model_2.joblib')

features_1 = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11]
df_features_1 = pd.DataFrame([features_1], columns=model_final_upr.feature_names_in_)
predict_2 = model_final_upr.predict(df_features_1)[0]

# st.text(f'Предсказание для модуля упругости при растяжении: {predict_2} ГПа')

# Вывод предсказания с жирным шрифтом для значения
st.markdown(
    f"""
    <div>
        Предсказание для модуля упругости при растяжении: <strong>{predict_2:.3f}</strong> ГПа
    </div>
    """,
    unsafe_allow_html=True
)
