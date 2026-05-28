import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    <div style="font-size: 34px; font-weight: 700; color: #1f2937;">
        Система прогнозирования итогового балла муниципальных образований
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
# Признаки
# ---------------------------------
sphere_columns = [f"sphere_{i}" for i in range(1, 9)]

sphere_names = {f"sphere_{i}": f"Сфера {i}" for i in range(1, 9)}

basic_feature_list = sphere_columns.copy()

extended_feature_list = basic_feature_list + [
    "quarter_num",
    "avg_sphere_score",
    "min_sphere_score",
    "max_sphere_score",
    "weak_spheres_count",
]

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("Параметры прогноза")

quarter_names_ru = {
    "Q1_2024": "1 кв.2024",
    "Q2_2024": "2 кв.2024",
    "Q3_2024": "3 кв.2024",
}

available_quarters = sorted(dataset["quarter"].unique())

selected_quarter_label = st.sidebar.selectbox(
    "Квартал",
    [quarter_names_ru[q] for q in available_quarters]
)

selected_quarter = {v: k for k, v in quarter_names_ru.items()}[selected_quarter_label]

available_mo = sorted(
    dataset[dataset["quarter"] == selected_quarter]["MO_code"].unique()
)

selected_mo = st.sidebar.selectbox("Муниципалитет", available_mo)

model_names = {
    "LinearRegression_basic": "Линейная регрессия",
    "RandomForest_extended": "Random Forest"
}

selected_model_label = st.sidebar.selectbox(
    "Модель",
    list(model_names.values())
)

selected_model_name = {v: k for k, v in model_names.items()}[selected_model_label]

# ---------------------------------
# Данные
# ---------------------------------
selected_row = dataset[
    (dataset["quarter"] == selected_quarter) &
    (dataset["MO_code"] == selected_mo)
].iloc[[0]]

actual_value = float(selected_row["total_score"].values[0])

# ---------------------------------
# Прогноз
# ---------------------------------
if selected_model_name == "LinearRegression_basic":
    scaler = StandardScaler()
    scaler.fit(dataset[basic_feature_list])

    X_input = scaler.transform(selected_row[basic_feature_list])

    predicted_value = float(linear_model.predict(X_input)[0])

    model_description = "Линейная модель"
else:
    predicted_value = float(
        random_forest_model.predict(selected_row[extended_feature_list])[0]
    )

    model_description = "Random Forest"

absolute_error = abs(actual_value - predicted_value)

# ---------------------------------
# Подготовка таблиц
# ---------------------------------
sphere_table = pd.DataFrame({
    "Сфера": [sphere_names[col] for col in sphere_columns],
    "Балл": [round(float(selected_row[col]), 2) for col in sphere_columns]
})

sphere_table_sorted = sphere_table.sort_values(by="Балл")

extra_features = ["quarter_num","avg_sphere_score","min_sphere_score","max_sphere_score","weak_spheres_count"]

extra_table = pd.DataFrame({
    "Признак": extra_features,
    "Значение": [round(float(selected_row[col]), 2) for col in extra_features]
})

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Прогноз", "📈 Аналитика", "ℹ️ О модели"])

# ---------- TAB 1 ----------
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("МО", selected_mo)
    col2.metric("Квартал", selected_quarter_label)
    col3.metric("Факт", f"{actual_value:.2f}")
    col4.metric("Прогноз", f"{predicted_value:.2f}")

    st.metric("Ошибка", f"{absolute_error:.2f}")

    fig, ax = plt.subplots()
    ax.bar(["Факт","Прогноз"], [actual_value, predicted_value])
    st.pyplot(fig)

# ---------- TAB 2 ----------
with tab2:
    st.subheader("Сферы")
    st.dataframe(sphere_table_sorted)

    fig, ax = plt.subplots()
    ax.bar(sphere_table_sorted["Сфера"], sphere_table_sorted["Балл"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Доп. признаки")
    st.dataframe(extra_table)

# ---------- TAB 3 ----------
with tab3:
    st.subheader("Модель")
    st.success(selected_model_label)
    st.write(model_description)

# ---------------------------------
st.caption("ML прогноз KPI")