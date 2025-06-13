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

def suggest_date_column(df):
    """Предполагает, какая колонка является датой, но не преобразует ее."""
    potential_date_cols = [
        'Datasales', 'datasales', 'date', 'Date', 'data', 'Дата', 'дата_продажи', 'timestamp'
    ]
    for col_name in potential_date_cols:
        if col_name in df.columns:
            return col_name
    # Если по имени не нашли, ищем по содержимому (но менее агрессивно)
    for col_name in df.select_dtypes(include=['object']).columns:
        try:
            # Проверяем, можно ли преобразовать хотя бы часть строк
            if pd.to_datetime(df[col_name], errors='coerce', infer_datetime_format=True).notna().sum() > 0:
                 # Проверяем, что в названии есть намек на дату
                if any(substr in col_name.lower() for substr in ['date', 'дата', 'day', 'день']):
                    return col_name
        except Exception:
            continue
    return None

def display_data_stats(total_rows, clean_rows, bad_date_rows, df_agg, date_col_name):
    with st.expander("📊 Смотрим на твои данные...", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Строк в файле 💅", f"{total_rows}")
        col2.metric("Строк для анализа 💎", f"{clean_rows}", help="Строки с заполненными 'Qty', 'Art', 'Magazin' и корректной датой.")
        col3.metric("Удалено из-за ошибок 🗑️", f"{(total_rows - clean_rows)}", help=f"Включая {bad_date_rows} строк с некорректным форматом даты в колонке '{date_col_name}'.")
        st.success(f"""
        - **Уникальных моделей очков:** {df_agg['Art'].nunique()} 🕶️
        - **Уникальных бутиков:** {df_agg['Magazin'].nunique()} 🏬
        - **Период продаж:** с {df_agg['first_sale_date'].min().strftime('%d.%m.%Y')} по {df_agg['first_sale_date'].max().strftime('%d.%m.%Y')} 🗓️
        """)

@st.cache_data
def process_and_aggregate(df, date_col_name):
    """Основная функция обработки данных, кэшируется."""
    initial_rows = len(df)
    
    # 1. Принудительное преобразование даты и удаление ошибок
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
    rows_before_date_drop = len(df)
    df.dropna(subset=[date_col_name], inplace=True)
    rows_after_date_drop = len(df)
    bad_date_rows = rows_before_date_drop - rows_after_date_drop

    # 2. Удаление пропусков в ключевых колонках
    crucial_cols = ['Qty', 'Art', 'Magazin']
    df_clean = df.dropna(subset=crucial_cols).copy()
    
    # 3. Агрегация
    df_clean = df_clean.sort_values(by=['Art', 'Magazin', date_col_name])
    series_of_first_dates = df_clean.groupby(['Art', 'Magazin'])[date_col_name].first()
    first_sale_dates = series_of_first_dates.reset_index(name='first_sale_date')
    df_merged = pd.merge(df_clean, first_sale_dates, on=['Art', 'Magazin'])
    df_30_days = df_merged[df_merged[date_col_name] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))].copy()
    agg_logic = {
        'Qty': 'sum', 'Sum': 'sum', 'Price': 'mean', 'Model': 'first', 'brand': 'first',
        'Segment': 'first', 'color': 'first', 'formaoprav': 'first', 'Sex': 'first', 'Metal-Plastic': 'first',
        'first_sale_date': 'first' # Добавляем для статистики
    }
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)

    stats = {
        "total_rows": initial_rows, 
        "clean_rows": len(df_agg), 
        "bad_date_rows": bad_date_rows
    }
    return df_agg, stats

@st.cache_resource
def train_model_with_optuna(_df_agg):
    # (Эта функция без изменений)
    target, cat_features = 'Qty_30_days', ['Magazin', 'brand', 'Segment', 'color', 'formaoprav', 'Sex', 'Metal-Plastic']
    df_processed = _df_agg.drop(columns=['Sum', 'Art', 'Model', 'first_sale_date'], errors='ignore')
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
    try:
        if dataset_file.name.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file)
        else:
            df_raw = pd.read_excel(dataset_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Не могу прочитать файл, дорогая! Ошибка: {e}")
        st.stop()
    
    # --- НОВЫЙ БЛОК: РУЧНОЙ ВЫБОР КОЛОНКИ С ДАТОЙ ---
    st.subheader("Шаг 1: Проверка данных 🧐")
    suggested_col = suggest_date_column(df_raw)
    all_columns = df_raw.columns.tolist()
    
    if suggested_col and suggested_col in all_columns:
        default_index = all_columns.index(suggested_col)
    else:
        default_index = 0
        
    date_col_name = st.selectbox(
        "🎀 Я думаю, колонка с датой это...",
        options=all_columns,
        index=default_index,
        help="Пожалуйста, выбери колонку, где указана дата продажи. Я постаралась угадать сама!"
    )
    
    df_agg, stats = process_and_aggregate(df_raw, date_col_name)
    display_data_stats(stats['total_rows'], stats['clean_rows'], stats['bad_date_rows'], df_agg, date_col_name)

    if df_agg is not None and not df_agg.empty:
        st.subheader("Шаг 2: Создание прогноза 🧙‍♀️")
        model, features, metrics = train_model_with_optuna(df_agg)
        if model:
            # (Остальная часть кода без изменений)
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
