import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------
# Настройка страницы
# ---------------------------------
st.set_page_config(
    page_title="KPI Forecast MVP",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------
# Заголовок
# ---------------------------------

st.markdown("""
<div style="
    background-color: #f5f7fb;
    padding: 24px 28px;
    border-radius: 14px;
    border: 1px solid #d9e2f0;
    margin-bottom: 20px;
">
    <div style="font-size: 18px; color: #4a5568; margin-bottom: 8px;">
        Прототип системы аналитики
    </div>
    <div style="font-size: 34px; font-weight: 700; color: #1f2937; line-height: 1.2; margin-bottom: 10px;">
        Система прогнозирования итогового балла муниципальных образований
    </div>
    <div style="font-size: 16px; color: #4b5563; line-height: 1.5;">
        Интерфейс предназначен для просмотра прогноза итогового балла муниципального образования
        на основе агрегированных квартальных показателей по 8 сферам.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Загрузка данных и моделей
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("data/processed/dataset_features_ready.xlsx")

@st.cache_resource
def load_models():
    linear_model = joblib.load("models/model_linear_regression_basic.pkl")
    random_forest_model = joblib.load("models/model_random_forest_extended.pkl")
    return linear_model, random_forest_model

dataset = load_data()
linear_model, random_forest_model = load_models()

# ---------------------------------
# Названия сфер
# ---------------------------------
sphere_columns = [
    "sphere_1",
    "sphere_2",
    "sphere_3",
    "sphere_4",
    "sphere_5",
    "sphere_6",
    "sphere_7",
    "sphere_8",
]

sphere_names = {
    "sphere_1": "Сфера 1",
    "sphere_2": "Сфера 2",
    "sphere_3": "Сфера 3",
    "sphere_4": "Сфера 4",
    "sphere_5": "Сфера 5",
    "sphere_6": "Сфера 6",
    "sphere_7": "Сфера 7",
    "sphere_8": "Сфера 8",
}

basic_feature_list = sphere_columns.copy()

extended_feature_list = basic_feature_list + [
    "quarter_num",
    "avg_sphere_score",
    "min_sphere_score",
    "max_sphere_score",
    "weak_spheres_count",
]

# ---------------------------------
# Боковая панель
# ---------------------------------
st.sidebar.header("Параметры прогноза")

quarter_names_ru = {
    "Q1_2024": "1 кв.2024",
    "Q2_2024": "2 кв.2024",
    "Q3_2024": "3 кв.2024",
}

available_quarters = sorted(dataset["quarter"].unique().tolist())

selected_quarter_label = st.sidebar.selectbox(
    "Выберите квартал",
    [quarter_names_ru[q] for q in available_quarters],
    index=len(available_quarters) - 1
)

selected_quarter = {
    value: key for key, value in quarter_names_ru.items()
}[selected_quarter_label]


available_municipalities = sorted(
    dataset[dataset["quarter"] == selected_quarter]["MO_code"].unique().tolist()
)

selected_mo = st.sidebar.selectbox(
    "Выберите муниципальное образование",
    available_municipalities
)

model_names_ru = {
    "LinearRegression_basic": "Линейная регрессия (базовая)",
    "RandomForest_extended": "Random Forest (расширенная)"
}

selected_model_label = st.sidebar.selectbox(
    "Выберите модель",
    list(model_names_ru.values()),
    index=1
)

selected_model_name = {
    value: key for key, value in model_names_ru.items()
}[selected_model_label]

# ---------------------------------
# Отбор данных по выбранному МО
# ---------------------------------
selected_row = dataset[
    (dataset["quarter"] == selected_quarter) &
    (dataset["MO_code"] == selected_mo)
].copy()

if selected_row.empty:
    st.error("Не удалось найти данные по выбранному муниципальному образованию.")
    st.stop()

selected_row = selected_row.iloc[[0]]

# ---------------------------------
# Прогноз
# ---------------------------------
actual_value = float(selected_row["total_score"].values[0])

if selected_model_name == "LinearRegression_basic":
    predicted_value = float(
        linear_model.predict(selected_row[basic_feature_list])[0]
    )
    model_description = (
        "Линейная регрессия на базовом наборе признаков. "
    )
else:
    predicted_value = float(
        random_forest_model.predict(selected_row[extended_feature_list])[0]
    )
    model_description = (
        "Random Forest на расширенном наборе признаков. "
    )

absolute_error = abs(actual_value - predicted_value)

# ---------------------------------
# Верхние карточки
# ---------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Муниципалитет", selected_mo)
col2.metric("Квартал", quarter_names_ru[selected_quarter])
col3.metric("Фактический балл", f"{actual_value:.2f}")
col4.metric("Прогноз", f"{predicted_value:.2f}")

st.metric("Абсолютная ошибка", f"{absolute_error:.2f}")

# ---------------------------------
# Описание модели
# ---------------------------------
st.subheader("Описание модели")
st.info(model_description)

# ---------------------------------
# Таблица по сферам
# ---------------------------------
st.subheader("Баллы по сферам")

sphere_table = pd.DataFrame({
    "Сфера": [sphere_names[col] for col in sphere_columns],
    "Балл": [round(float(selected_row[col].values[0]), 2) for col in sphere_columns]
})

st.dataframe(sphere_table, use_container_width=True)

# ---------------------------------
# График по сферам
# ---------------------------------
st.subheader("Визуализация значений по сферам")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(sphere_table["Сфера"], sphere_table["Балл"])
ax.set_title("Баллы по сферам")
ax.set_xlabel("Сфера")
ax.set_ylabel("Балл")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------------------------
# Дополнительные признаки
# ---------------------------------
st.subheader("Дополнительные признаки")

feature_names_ru = {
    "quarter_num": "Номер квартала",
    "avg_sphere_score": "Средний балл по сферам",
    "min_sphere_score": "Минимальный балл по сферам",
    "max_sphere_score": "Максимальный балл по сферам",
    "weak_spheres_count": "Количество сфер с низким баллом",
}

extra_feature_columns = [
    "quarter_num",
    "avg_sphere_score",
    "min_sphere_score",
    "max_sphere_score",
    "weak_spheres_count",
]

extra_features_table = pd.DataFrame({
    "Признак": [feature_names_ru[col] for col in extra_feature_columns],
    "Значение": [
    round(float(selected_row["quarter_num"].values[0]), 2),
    round(float(selected_row["avg_sphere_score"].values[0]), 2),
    round(float(selected_row["min_sphere_score"].values[0]), 2),
    round(float(selected_row["max_sphere_score"].values[0]), 2),
    round(float(selected_row["weak_spheres_count"].values[0]), 2),
]
})

st.dataframe(extra_features_table, use_container_width=True)

# ---------------------------------
# Нижний комментарий
# ---------------------------------
st.markdown("---")
st.caption(
    "Приложение использует обезличенные квартальные данные, агрегированные показатели "
    "по 8 сферам и обученные модели машинного обучения."
)