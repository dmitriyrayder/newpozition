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
.main {
    background-color: #fce4ec; /* Нежно-розовый */
}
/* Заголовок H1 */
h1 {
    font-family: 'Comic Sans MS', cursive, sans-serif;
    color: #e91e63; /* Ярко-розовый */
    text-align: center;
    text-shadow: 2px 2px 4px #f8bbd0;
}
/* Подзаголовки H2, H3 */
h2, h3 {
    font-family: 'Comic Sans MS', cursive, sans-serif;
    color: #ad1457; /* Глубокий розовый */
}
/* Стилизация кнопки */
.stButton>button {
    color: white;
    background-image: linear-gradient(to right, #f06292, #e91e63);
    border-radius: 25px;
    border: 2px solid #ad1457;
    padding: 12px 28px;
    font-weight: bold;
    font-size: 18px;
    box-shadow: 0 4px 15px 0 rgba(233, 30, 99, 0.4);
    transition: all 0.3s ease 0s;
}
.stButton>button:hover {
    background-position: right center;
    box-shadow: 0 4px 15px 0 rgba(233, 30, 99, 0.75);
}
/* Стилизация контейнера Expander */
.stExpander {
    border: 2px solid #f8bbd0;
    border-radius: 10px;
    background-color: #fff1f8;
}
</style>
""", unsafe_allow_html=True)

st.title("💖 Модный Советник по Продажам 💖")

# --- БЛОК ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

def find_and_convert_date_column(df):
    potential_date_cols = ['Datasales', 'datasales', 'date', 'Date', 'Дата', 'дата_продажи', 'timestamp']
    for col_name in potential_date_cols:
        if col_name in df.columns:
            st.info(f"Найдена колонка с датой: '{col_name}'. Преобразую... ✨")
            try:
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                if df[col_name].notna().sum() / len(df) > 0.8: return df, col_name
            except Exception: continue
    
    st.warning("Не нашла колонку с датой по имени, ищу по содержимому...")
    for col_name in df.select_dtypes(include=['object']).columns:
        try:
            converted_col = pd.to_datetime(df[col_name], errors='coerce')
            if converted_col.notna().sum() / len(df) > 0.8:
                st.success(f"Нашла! ✨ Колонка с датой: '{col_name}'.")
                df[col_name] = converted_col
                return df, col_name
        except Exception: continue
    return df, None

def display_data_stats(df_raw, df_clean, date_col_name):
    with st.expander("📊 Смотрим на твои данные...", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Строк в файле 💅", f"{len(df_raw)}")
        col2.metric("Строк для анализа 💎", f"{len(df_clean)}")
        col3.metric("Удалено лишних 🗑️", f"{len(df_raw) - len(df_clean)}")
        st.success(f"""
        - **Уникальных моделей очков:** {df_clean['Art'].nunique()} 🕶️
        - **Уникальных бутиков:** {df_clean['Magazin'].nunique()} 🏬
        - **Период продаж:** с {df_clean[date_col_name].min().strftime('%d.%m.%Y')} по {df_clean[date_col_name].max().strftime('%d.%m.%Y')} 🗓️
        """)

@st.cache_data
def load_and_prepare_data(uploaded_file):
    try:
        # Чтение CSV или Excel
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Ой! Я не знаю такой формат файла. Попробуй .csv, .xlsx или .xls 🎀")
            return None
    except Exception as e:
        st.error(f"Не могу прочитать файл, дорогая! Ошибка: {e}")
        return None

    df, date_col_name = find_and_convert_date_column(df_raw.copy())
    if not date_col_name:
        st.error("Не нашла колонку с датой в файле. Проверь, пожалуйста! 🥺")
        return None
    
    crucial_cols = ['Qty', 'Art', 'Magazin', date_col_name]
    df_clean = df.dropna(subset=crucial_cols).copy()
    display_data_stats(df_raw, df_clean, date_col_name)

    st.info("Агрегирую продажи за первые 30 дней... Магия в процессе! ✨")
    df_clean = df_clean.sort_values(by=['Art', 'Magazin', date_col_name])
    
    # --- ИСПРАВЛЕННЫЙ БЛОК ДЛЯ ИЗБЕЖАНИЯ ОШИБКИ ---
    # Получаем серию с первыми датами
    series_of_first_dates = df_clean.groupby(['Art', 'Magazin'])[date_col_name].first()
    # Безопасно сбрасываем индекс, сразу давая столбцу с датами новое имя 'first_sale_date'
    first_sale_dates = series_of_first_dates.reset_index(name='first_sale_date')
    # --- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА ---
    
    df_merged = pd.merge(df_clean, first_sale_dates, on=['Art', 'Magazin'])
    df_30_days = df_merged[df_merged[date_col_name] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))].copy()

    agg_logic = {
        'Qty': 'sum', 'Sum': 'sum', 'Price': 'mean', 'Model': 'first', 'brand': 'first', 
        'Segment': 'first', 'color': 'first', 'formaoprav': 'first', 'Sex': 'first', 'Metal-Plastic': 'first'
    }
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)
    return df_agg

@st.cache_resource
def train_model_with_optuna(_df_agg):
    target, cat_features = 'Qty_30_days', ['Magazin', 'brand', 'Segment', 'color', 'formaoprav', 'Sex', 'Metal-Plastic']
    df_processed = _df_agg.drop(columns=['Sum', 'Art', 'Model'], errors='ignore')
    features = [col for col in df_processed.columns if col != target]
    for col in cat_features:
        if col in df_processed.columns: df_processed[col] = df_processed[col].astype(str)

    X, y = df_processed[features], df_processed[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with st.spinner("🔮 Подбираю лучшие волшебные параметры для модели..."):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: mean_absolute_error(y_test, CatBoostRegressor(
            iterations=1000, learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            depth=trial.suggest_int('depth', 4, 10), verbose=0, random_seed=42
        ).fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True).predict(X_test)), n_trials=30)
    
    st.success(f"Волшебство сработало! 💫 Лучшие параметры: {study.best_params}")
    final_model = CatBoostRegressor(**study.best_params, iterations=1500, verbose=0, random_seed=42).fit(X, y, cat_features=cat_features)
    test_preds = final_model.predict(X_test)
    return final_model, features, {'MAE': mean_absolute_error(y_test, test_preds), 'R2': r2_score(y_test, test_preds)}

# --- ОСНОВНОЙ БЛОК STREAMLIT ---

dataset_file = st.file_uploader("💖 Загрузи свой файл с продажами (.xlsx, .xls, .csv)", type=["csv", "xlsx", "xls"])

if dataset_file:
    df_agg = load_and_prepare_data(dataset_file)
    if df_agg is not None and not df_agg.empty:
        model, features, metrics = train_model_with_optuna(df_agg)
        if model:
            st.header("📊 Оценка моей работы")
            col1, col2 = st.columns(2)
            col1.metric("Средняя ошибка (MAE)", f"{metrics['MAE']:.2f} шт.", "+/- столько я могу ошибиться")
            col2.metric("Точность предсказаний (R²)", f"{metrics['R2']:.2%}", "чем ближе к 100%, тем лучше!")

            st.header("✍️ Опиши свою новую блестящую модель очков")
            with st.form("product_form"):
                col1, col2 = st.columns(2)
                with col1:
                    brand = st.text_input("Бренд 👑", help="Например, Miu Miu")
                    forma = st.text_input("Форма оправы 👓", help="Например, Кошачий глаз")
                    sex = st.selectbox("Для кого? 👠", df_agg['Sex'].unique())
                    price = st.number_input("Цена 💰", min_value=0.0, step=100.0, format="%.2f")
                with col2:
                    segment = st.selectbox("Сегмент 💅", df_agg['Segment'].unique())
                    color = st.text_input("Цвет 🌈", help="Например, Розовый")
                    material = st.selectbox("Материал ✨", df_agg['Metal-Plastic'].unique())
                submitted = st.form_submit_button("Найти лучшие бутики! 🚀")

            if submitted:
                magaziny = df_agg['Magazin'].unique()
                new_product_data = {'brand': brand, 'Segment': segment, 'color': color, 'formaoprav': forma,
                                    'Sex': sex, 'Metal-Plastic': material, 'Price': price}
                recs_df = pd.DataFrame([dict(item, Magazin=mag) for mag in magaziny for item in [new_product_data]])[features]
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
    st.info("💋 Привет! Загрузи файлик, и я помогу тебе стать звездой продаж!")
