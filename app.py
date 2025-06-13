import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

# --- СТИЛИЗАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Модный Советник", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Основной фон приложения */
.main { background-color: #fce4ec; }
h1 { font-family: 'Comic Sans MS', cursive, sans-serif; color: #e91e63; text-align: center; text-shadow: 2px 2px 4px #f8bbd0; }
h2, h3 { font-family: 'Comic Sans MS', cursive, sans-serif; color: #ad1457; }
.stButton>button {
    color: white; background-image: linear-gradient(to right, #f06292, #e91e63); border-radius: 25px;
    border: 2px solid #ad1457; padding: 12px 28px; font-weight: bold; font-size: 18px;
    box-shadow: 0 4px 15px 0 rgba(233, 30, 99, 0.4); transition: all 0.3s ease 0s;
}
.stButton>button:hover { background-position: right center; box-shadow: 0 4px 15px 0 rgba(233, 30, 99, 0.75); }
.stExpander { border: 2px solid #f8bbd0; border-radius: 10px; background-color: #fff1f8; }
</style>
""", unsafe_allow_html=True)

st.title("💖 Модный Советник по Продажам 💖")

# --- БЛОК ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

@st.cache_data
def process_and_aggregate(_df, column_map):
    """Основная функция обработки данных, кэшируется."""
    df = _df.copy()
    
    # Переименовываем колонки в стандартные имена для удобства
    internal_map = {v: k for k, v in column_map.items()}
    df.rename(columns=internal_map, inplace=True)
    
    initial_rows = len(df)
    
    # 1. Обработка дат
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    bad_date_rows = df['date'].isna().sum()
    df.dropna(subset=['date'], inplace=True)

    # 2. Удаление пропусков в ключевых колонках
    crucial_cols = ['Qty', 'Art', 'Magazin', 'Price']
    df.dropna(subset=crucial_cols, inplace=True)
    
    # 3. Агрегация
    df = df.sort_values(by=['Art', 'Magazin', 'date'])
    series_of_first_dates = df.groupby(['Art', 'Magazin'])['date'].first()
    first_sale_dates = series_of_first_dates.reset_index(name='first_sale_date')
    df_merged = pd.merge(df, first_sale_dates, on=['Art', 'Magazin'])
    df_30_days = df_merged[df_merged['date'] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))].copy()

    agg_logic = {'Qty': 'sum', 'Price': 'mean'}
    # Динамически добавляем агрегацию для категориальных признаков
    for cat_col in column_map['categorical_features']:
        agg_logic[cat_col] = 'first'
    
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)
    
    stats = {
        "total_rows": initial_rows, 
        "final_rows": len(df_agg), 
        "bad_date_rows": bad_date_rows
    }
    return df_agg, stats

@st.cache_resource
def train_model_with_optuna(_df_agg, cat_features):
    target = 'Qty_30_days'
    # 'Art' никогда не используется как признак
    features = ['Magazin', 'Price'] + cat_features
    df_processed = _df_agg[features + [target]]
    
    for col in cat_features + ['Magazin']:
        df_processed[col] = df_processed[col].astype(str)

    X, y = df_processed[features], df_processed[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with st.spinner("🔮 Подбираю лучшие волшебные параметры для модели..."):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: mean_absolute_error(y_test, CatBoostRegressor(
            iterations=1000, learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            depth=trial.suggest_int('depth', 4, 10), verbose=0, random_seed=42
        ).fit(X_train, y_train, cat_features=cat_features + ['Magazin'], eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True).predict(X_test)), n_trials=30)
    
    st.success(f"Волшебство сработало! 💫 Лучшие параметры: {study.best_params}")
    final_model = CatBoostRegressor(**study.best_params, iterations=1500, verbose=0, random_seed=42).fit(X, y, cat_features=cat_features + ['Magazin'])
    test_preds = final_model.predict(X_test)
    return final_model, features, {'MAE': mean_absolute_error(y_test, test_preds), 'R2': r2_score(y_test, test_preds)}

# --- ОСНОВНОЙ БЛОК STREAMLIT ---

if 'processed' not in st.session_state:
    st.session_state.processed = False

dataset_file = st.file_uploader("💖 Загрузи свой файл с продажами (.xlsx, .xls, .csv)", type=["csv", "xlsx", "xls"])

if dataset_file:
    try:
        if dataset_file.name.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file)
        else:
            df_raw = pd.read_excel(dataset_file, engine='openpyxl')
        st.session_state.df_raw = df_raw
    except Exception as e:
        st.error(f"Не могу прочитать файл, дорогая! Ошибка: {e}")
        st.stop()
    
    st.subheader("Шаг 1: Помоги мне понять твои данные 🧐")
    st.info("Пожалуйста, укажи, какие колонки в твоем файле за что отвечают. Это очень важно! 🙏")
    
    all_columns = st.session_state.df_raw.columns.tolist()
    
    with st.form("mapping_form"):
        col1, col2 = st.columns(2)
        with col1:
            art_col = st.selectbox("Артикул товара (ID)", all_columns, index=0)
            magazin_col = st.selectbox("Название магазина", all_columns, index=1)
            date_col = st.selectbox("Дата продажи", all_columns, index=2)
        with col2:
            qty_col = st.selectbox("Количество проданного товара (шт.)", all_columns, index=3)
            price_col = st.selectbox("Цена товара", all_columns, index=4)
        
        available_features = [c for c in all_columns if c not in [art_col, magazin_col, date_col, qty_col, price_col]]
        cat_features_selected = st.multiselect(
            "Описательные характеристики товара (выбери все, что описывает товар)",
            available_features,
            help="Выбери колонки вроде 'Бренд', 'Цвет', 'Материал', 'Сегмент' и т.д."
        )
        
        submitted_mapping = st.form_submit_button("Обработать данные и обучить модель 🚀")

    if submitted_mapping:
        column_map = {
            'Art': art_col, 'Magazin': magazin_col, 'date': date_col,
            'Qty': qty_col, 'Price': price_col,
            'categorical_features': cat_features_selected
        }
        st.session_state.column_map = column_map
        
        df_agg, stats = process_and_aggregate(st.session_state.df_raw, column_map)
        
        with st.expander("📊 Смотрим на твои данные...", expanded=True):
            st.metric("Строк в файле 💅", f"{stats['total_rows']}")
            st.metric("Строк после очистки и агрегации 💎", f"{stats['final_rows']}")
            st.metric("Строк с плохой датой 🗑️", f"{stats['bad_date_rows']}")
        
        st.session_state.df_agg = df_agg
        
        model, features, metrics = train_model_with_optuna(df_agg, cat_features_selected)
        st.session_state.model = model
        st.session_state.features = features
        st.session_state.metrics = metrics
        st.session_state.processed = True
        st.rerun()

if st.session_state.processed:
    st.subheader("Шаг 2: Создание прогноза 🧙‍♀️")
    st.header("📊 Оценка моей работы")
    metrics = st.session_state.metrics
    col1, col2 = st.columns(2)
    col1.metric("Средняя ошибка (MAE)", f"{metrics['MAE']:.2f} шт.", "+/- столько я могу ошибиться")
    col2.metric("Точность предсказаний (R²)", f"{metrics['R2']:.2%}", "чем ближе к 100%, тем лучше!")

    st.header("✍️ Опиши свою новую блестящую модель очков")
    with st.form("product_form"):
        new_product_data = {}
        # Динамически создаем поля для ввода
        for feature in st.session_state.column_map['categorical_features']:
            unique_vals = st.session_state.df_agg[feature].dropna().unique().tolist()
            new_product_data[feature] = st.selectbox(f"{feature} ✨", unique_vals)
        
        new_product_data['Price'] = st.number_input("Цена 💰", min_value=0.0, step=100.0, format="%.2f")
        
        submitted_prediction = st.form_submit_button("Найти лучшие бутики! 🚀")

    if submitted_prediction:
        df_agg = st.session_state.df_agg
        model = st.session_state.model
        features = st.session_state.features
        
        magaziny = df_agg['Magazin'].unique()
        
        recs_list = []
        for magazin in magaziny:
            row = new_product_data.copy()
            row['Magazin'] = magazin
            recs_list.append(row)

        recs_df = pd.DataFrame(recs_list)[features]
        recs_df['Pred_Qty_30_days'] = np.maximum(0, model.predict(recs_df).round(0))
        max_pred = recs_df['Pred_Qty_30_days'].max()
        recs_df['Rating_%'] = (recs_df['Pred_Qty_30_days'] / max_pred * 100).round(0) if max_pred > 0 else 0
        
        top_magaziny = recs_df.sort_values(by='Pred_Qty_30_days', ascending=False).rename(columns={
            'Magazin': 'Бутик', 'Pred_Qty_30_days': 'Прогноз продаж (30 дней, шт.)', 'Rating_%': 'Рейтинг успеха (%)'
        })

        st.subheader("🎉 Вот лучшие места для твоей новинки! 🎉")
        st.dataframe(top_magaziny[['Бутик', 'Прогноз продаж (30 дней, шт.)', 'Рейтинг успеха (%)']].style.highlight_max(
            subset=['Прогноз продаж (30 дней, шт.)'], color='#f8bbd0', axis=0
        ).format({'Прогноз продаж (30 дней, шт.)': '{:.0f}', 'Рейтинг успеха (%)': '{:.0f}%'}), use_container_width=True)
else:
    if not dataset_file:
        st.info("💋 Привет! Загрузи файлик, а потом помоги мне понять, где какие данные.")
