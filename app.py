import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

# Отключаем детальное логирование Optuna в консоли
optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(page_title="Рекомендатор магазинов", layout="wide")

st.title("🎯 Рекомендатор магазинов для нового товара")

# --- БЛОК ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

def find_and_convert_date_column(df):
    """
    Автоматически находит и конвертирует колонку с датами.
    Сначала ищет по популярным именам, затем пытается конвертировать object-колонки.
    """
    potential_date_cols = ['Datasales', 'datasales', 'date', 'Date', 'Дата', 'дата_продажи', 'timestamp']
    
    # Поиск по имени
    for col_name in potential_date_cols:
        if col_name in df.columns:
            st.info(f"Найдена колонка с датой: '{col_name}'. Попытка конвертации...")
            try:
                # errors='coerce' превратит неудачные парсинги в NaT (Not a Time)
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                # Если после конвертации почти все значения валидны, считаем это успехом
                if df[col_name].notna().sum() / len(df) > 0.8:
                    return df, col_name
            except Exception:
                continue # Попробуем следующую

    # Если по имени не нашли, ищем по содержимому
    st.warning("Не найдено стандартное имя колонки с датой. Пытаюсь найти по содержимому...")
    for col_name in df.select_dtypes(include=['object']).columns:
        try:
            converted_col = pd.to_datetime(df[col_name], errors='coerce')
            # Если успешно конвертировано более 80% строк, считаем колонку датой
            if converted_col.notna().sum() / len(df) > 0.8:
                st.info(f"Автоматически определена колонка с датой: '{col_name}'.")
                df[col_name] = converted_col
                return df, col_name
        except Exception:
            continue
            
    return df, None

def display_data_stats(df_raw, df_clean, date_col_name):
    """Показывает детальную статистику по загруженному датасету."""
    with st.expander("🔍 Статистика по загруженным данным", expanded=True):
        initial_rows = len(df_raw)
        clean_rows = len(df_clean)
        dropped_rows = initial_rows - clean_rows
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Строк в файле", f"{initial_rows}")
        col2.metric("Строк после очистки", f"{clean_rows}", help="Удалены строки с пропусками в 'Qty', 'Art', 'Magazin' или дате.")
        col3.metric("Удалено строк", f"{dropped_rows}")
        
        st.info(f"""
        - **Уникальных товаров (Art):** {df_clean['Art'].nunique()}
        - **Уникальных магазинов (Magazin):** {df_clean['Magazin'].nunique()}
        - **Период данных:** с {df_clean[date_col_name].min().strftime('%d.%m.%Y')} по {df_clean[date_col_name].max().strftime('%d.%m.%Y')}
        """)

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Загружает, проверяет, очищает и агрегирует данные."""
    try:
        df_raw = pd.read_xlsx(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return None

    # 1. АВТОМАТИЧЕСКИЙ ПОИСК И КОНВЕРТАЦИЯ ДАТЫ
    df, date_col_name = find_and_convert_date_column(df_raw.copy())
    if not date_col_name:
        st.error("Не удалось найти и обработать колонку с датой в вашем файле. Пожалуйста, проверьте данные.")
        return None
    
    # 2. ОЧИСТКА ДАННЫХ
    crucial_cols = ['Qty', 'Art', 'Magazin', date_col_name]
    df_clean = df.dropna(subset=crucial_cols).copy()
    
    # 3. ОТОБРАЖЕНИЕ СТАТИСТИКИ
    display_data_stats(df_raw, df_clean, date_col_name)

    # 4. АГРЕГАЦИЯ ДАННЫХ (ПРОДАЖИ ЗА ПЕРВЫЕ 30 ДНЕЙ)
    st.info("Агрегирую продажи за первые 30 дней для каждого товара в каждом магазине...")
    df_clean = df_clean.sort_values(by=['Art', 'Magazin', date_col_name])
    first_sale_dates = df_clean.groupby(['Art', 'Magazin'])[date_col_name].first().reset_index()
    first_sale_dates.rename(columns={date_col_name: 'first_sale_date'}, inplace=True)
    
    df_merged = pd.merge(df_clean, first_sale_dates, on=['Art', 'Magazin'])
    df_30_days = df_merged[df_merged[date_col_name] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))].copy()

    agg_logic = {
        'Qty': 'sum', 'Sum': 'sum', 'Price': 'mean', 'Model': 'first',
        'brand': 'first', 'Segment': 'first', 'color': 'first',
        'formaoprav': 'first', 'Sex': 'first', 'Metal-Plastic': 'first'
    }
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)
    
    return df_agg

@st.cache_resource
def train_model_with_optuna(_df_agg):
    """Проводит подбор гиперпараметров и обучает финальную модель."""
    target = 'Qty_30_days'
    cat_features = ['Magazin', 'brand', 'Segment', 'color', 'formaoprav', 'Sex', 'Metal-Plastic']
    drop_cols = ['Sum', 'Art', 'Model'] 
    
    df_processed = _df_agg.drop(columns=drop_cols, errors='ignore')
    features = [col for col in df_processed.columns if col != target]
    
    for col in cat_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    X = df_processed[features]
    y = df_processed[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    st.info("Запускаю автоподбор параметров модели с помощью Optuna...")
    
    def objective(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'verbose': 0, 'random_seed': 42
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True)
        return mean_absolute_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params
    st.success(f"Лучшие параметры найдены: {best_params}")

    st.info("Обучаю финальную модель на всех данных...")
    final_model = CatBoostRegressor(**best_params, iterations=1500, verbose=0, random_seed=42)
    final_model.fit(X, y, cat_features=cat_features)

    test_preds = final_model.predict(X_test)
    metrics = {'MAE': mean_absolute_error(y_test, test_preds), 'R2': r2_score(y_test, test_preds)}
    
    return final_model, features, cat_features, metrics

# --- ОСНОВНОЙ БЛОК STREAMLIT ---

dataset_file = st.file_uploader("\U0001F4C2 Загрузите xlsx-файл с данными о продажах", type=["xlsx"])

if dataset_file:
    df_agg = load_and_prepare_data(dataset_file)
    
    if df_agg is not None and not df_agg.empty:
        model, features, cat_features, metrics = train_model_with_optuna(df_agg)

        if model:
            st.header("📊 Оценка качества модели")
            col1, col2 = st.columns(2)
            col1.metric("Средняя абсолютная ошибка (MAE)", f"{metrics['MAE']:.2f} шт.")
            col2.metric("Коэффициент детерминации (R²)", f"{metrics['R2']:.2%}")
            st.caption("MAE показывает, на сколько штук в среднем ошибается прогноз за 30 дней. R² показывает, какую долю изменчивости данных объясняет модель.")

            st.header("✍️ Введите характеристики нового товара")
            with st.form("product_form"):
                col1, col2 = st.columns(2)
                with col1:
                    brand = st.text_input("Brand (бренд)", help="Например, Ray-Ban")
                    forma = st.text_input("Forma oprav (форма оправы)", help="Например, Авиатор")
                    sex = st.selectbox("Sex (пол)", df_agg['Sex'].unique())
                    price = st.number_input("Price (цена)", min_value=0.0, step=100.0, format="%.2f")
                with col2:
                    segment = st.selectbox("Segment (сегмент)", df_agg['Segment'].unique())
                    color = st.text_input("Color (цвет)", help="Например, Черный")
                    material = st.selectbox("Metal-Plastic (материал)", df_agg['Metal-Plastic'].unique())
                
                submitted = st.form_submit_button("🚀 Получить рекомендации")

            if submitted:
                magaziny = df_agg['Magazin'].unique()
                new_product_data = {'brand': brand, 'Segment': segment, 'color': color, 'formaoprav': forma,
                                    'Sex': sex, 'Metal-Plastic': material, 'Price': price}
                
                recs_list = [dict(item, Magazin=mag) for mag in magaziny for item in [new_product_data]]
                recs_df = pd.DataFrame(recs_list)[features]

                recs_df['Pred_Qty_30_days'] = np.maximum(0, model.predict(recs_df).round(0))
                
                max_pred = recs_df['Pred_Qty_30_days'].max()
                recs_df['Rating_%'] = (recs_df['Pred_Qty_30_days'] / max_pred * 100).round(0) if max_pred > 0 else 0
                
                top_magaziny = recs_df.sort_values(by='Pred_Qty_30_days', ascending=False).reset_index(drop=True)

                st.subheader("\U0001F4C8 Рекомендованные магазины для нового товара")
                st.table(top_magaziny[['Magazin', 'Pred_Qty_30_days', 'Rating_%']].rename(columns={
                    'Magazin': 'Магазин', 'Pred_Qty_30_days': 'Прогноз продаж (30 дней, шт.)', 'Rating_%': 'Рейтинг (%)'
                }))
else:
    st.info("Пожалуйста, загрузите CSV-файл с данными о продажах для начала работы.")
