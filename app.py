import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna
import plotly.express as px
import re

# ==============================================================================
# 1. СТРУКТУРА ИНТЕРФЕЙСА И СТИЛИ
# ==============================================================================
st.set_page_config(page_title="Модный Советник", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Основной фон и шрифты */
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"; }
.main { background-color: #f0f2f6; }
/* Стилизация кнопок */
.stButton>button {
    border-radius: 8px; border: 2px solid #ff4b4b; color: #ff4b4b;
    background-color: transparent; font-weight: bold; transition: all 0.3s;
}
.stButton>button:hover {
    border-color: #ff4b4b; color: white; background-color: #ff4b4b;
}
/* Стиль для основной кнопки (type="primary") */
div[data-testid="stForm"] .stButton>button[kind="primary"] {
    border-color: #ff4b4b; color: white; background-color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 8. ТЕХНИЧЕСКАЯ ЧАСТЬ: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def auto_detect_column(columns, keywords, default_index=0):
    """Автоматическое определение индекса колонки по ключевым словам."""
    for keyword in keywords:
        for i, col in enumerate(columns):
            if keyword.lower() in col.lower():
                return i
    # Если ничего не найдено, возвращаем default_index, но не больше длины списка
    return min(default_index, len(columns) - 1 if columns else 0)

def extract_features_from_description(descriptions):
    """Упрощенное извлечение признаков из текстового описания."""
    features = pd.DataFrame(index=descriptions.index)
    
    # Словарь: имя нового признака -> [ключевые слова для поиска]
    extraction_map = {
        'brand_extracted': ['ray-ban', 'oakley', 'gucci', 'prada', 'polaroid'],
        'material_extracted': ['металл', 'пластик', 'дерево', 'комбинированный'],
        'shape_extracted': ['авиатор', 'вайфарер', 'круглые', 'кошачий глаз']
    }
    
    for feature_name, keywords in extraction_map.items():
        # Создаем паттерн RegEx, который найдет любое из ключевых слов
        pattern = re.compile(f'({"|".join(keywords)})', re.IGNORECASE)
        features[feature_name] = descriptions.str.findall(pattern).str[0].str.lower().fillna('не определен')
        
    return features

@st.cache_data
def process_data_and_train(_df, column_map, feature_config):
    """
    Основная функция: обрабатывает данные, извлекает признаки, обучает модель.
    """
    df = _df.copy()
    # Переименовываем основные колонки для удобства
    df.rename(columns={v: k for k, v in column_map.items() if v}, inplace=True)

    # 1. Валидация и очистка
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'Art', 'Magazin', 'Qty', 'Price'], inplace=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)

    # 2. Извлечение признаков
    all_features_df = pd.DataFrame(index=df.index)
    
    # 2.1 Автоматическое извлечение из описания
    if feature_config['describe_col'] != "Не использовать":
        extracted = extract_features_from_description(df['Describe'])
        all_features_df = pd.concat([all_features_df, extracted], axis=1)

    # 2.2 Добавление признаков из выбранных колонок
    for feature, source_col in feature_config['manual_features'].items():
        if source_col and source_col in df.columns:
            all_features_df[feature] = df[source_col].astype(str).fillna('не определен')

    # 3. Агрегация данных
    df_with_features = pd.concat([df[['Art', 'Magazin', 'date', 'Qty', 'Price']], all_features_df], axis=1)
    df_with_features = df_with_features.sort_values(by=['Art', 'Magazin', 'date'])
    first_sale_dates = df_with_features.groupby(['Art', 'Magazin'])['date'].first().reset_index(name='first_sale_date')
    df_merged = pd.merge(df_with_features, first_sale_dates, on=['Art', 'Magazin'])
    df_30_days = df_merged[df_merged['date'] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))].copy()

    agg_logic = {'Qty': 'sum', 'Price': 'mean'}
    feature_cols = [col for col in all_features_df.columns if col in df_30_days.columns]
    for col in feature_cols:
        agg_logic[col] = 'first'
    
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)

    # 4. Обучение модели
    if len(df_agg) < 50: # Проверка на достаточность данных
        return None, None, None, "Слишком мало данных для обучения после обработки."

    target = 'Qty_30_days'
    cat_features_to_use = ['Magazin'] + feature_cols
    features_to_use = ['Price'] + cat_features_to_use
    
    X = df_agg[features_to_use]
    y = df_agg[target]
    
    for col in cat_features_to_use:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optuna
    def objective(trial):
        params = {
            'iterations': 500, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 8), 'verbose': 0, 'random_seed': 42
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_features_to_use, eval_set=(X_test, y_test), early_stopping_rounds=30, use_best_model=True)
        return mean_absolute_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction='minimize')
    with st.spinner("Оптимизирую модель..."):
        study.optimize(objective, n_trials=20)
    
    final_model = CatBoostRegressor(**study.best_params, iterations=1000, verbose=0, random_seed=42)
    final_model.fit(X, y, cat_features=cat_features_to_use)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, final_model.predict(X_test)),
        'R2': r2_score(y_test, final_model.predict(X_test))
    }
    
    return final_model, features_to_use, metrics, None

# ==============================================================================
# ОСНОВНОЙ КОД ПРИЛОЖЕНИЯ
# ==============================================================================

st.title("💖 Модный Советник по Продажам")

# Инициализация состояния
if 'step' not in st.session_state:
    st.session_state.step = 1

# --- Блок 1: Загрузка файла в боковой панели ---
with st.sidebar:
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл Excel или CSV", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df_raw = df
            st.success("Файл успешно загружен!")
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()
    
if 'df_raw' in st.session_state:
    df_raw = st.session_state.df_raw
    available_columns = df_raw.columns.tolist()

    # --- Блок 2: Настройка колонок ---
    st.header("2. Настройка датасета")
    with st.form("column_mapping_form"):
        st.subheader("🎯 Сопоставление основных колонок")
        st.caption("Помогите мне понять, где какие данные находятся. Я попробую угадать сама!")
        c1, c2 = st.columns(2)
        with c1:
            col_magazin = st.selectbox("Колонка МАГАЗИН:", available_columns, index=auto_detect_column(available_columns, ['magazin', 'магазин', 'store']))
            col_art = st.selectbox("Колонка АРТИКУЛ:", available_columns, index=auto_detect_column(available_columns, ['art', 'артикул', 'sku'], 1))
            col_qty = st.selectbox("Колонка КОЛИЧЕСТВО:", available_columns, index=auto_detect_column(available_columns, ['qty', 'количество'], 2))
        with c2:
            col_date = st.selectbox("Колонка ДАТА ПРОДАЖИ:", available_columns, index=auto_detect_column(available_columns, ['datasales', 'дата'], 3))
            col_price = st.selectbox("Колонка ЦЕНА:", available_columns, index=auto_detect_column(available_columns, ['price', 'цена'], 4))
            col_describe = st.selectbox("Колонка ОПИСАНИЕ ТОВАРА:", available_columns + ["Не использовать"], index=auto_detect_column(available_columns, ['describe', 'описание'], 5))

        # --- Блок 3: Настройка признаков ---
        st.subheader("✋ Ручная настройка признаков товара")
        st.caption("Какие еще колонки содержат важную информацию о товаре? (например, Бренд, Цвет, Пол)")
        other_feature_cols = st.multiselect(
            "Выберите дополнительные колонки-признаки:",
            [c for c in available_columns if c not in [col_magazin, col_art, col_qty, col_date, col_price, col_describe]]
        )
        
        submitted = st.form_submit_button("✅ Подтвердить и обучить модель", type="primary")

    if submitted:
        st.session_state.step = 2
        column_map = {'Magazin': col_magazin, 'Art': col_art, 'date': col_date, 'Qty': col_qty, 'Price': col_price}
        
        # Собираем конфигурацию признаков
        feature_config = {
            'describe_col': col_describe,
            'manual_features': {col: col for col in other_feature_cols}
        }

        # Запускаем обработку и обучение
        model, features, metrics, error_msg = process_data_and_train(df_raw, column_map, feature_config)

        if error_msg:
            st.error(error_msg)
            st.stop()
            
        st.session_state.model = model
        st.session_state.features = features
        st.session_state.metrics = metrics
        st.session_state.df_agg = df_agg # Сохраняем агрегированные данные для профилирования
        st.session_state.feature_config = feature_config # Сохраняем для интерфейса ввода

# --- Блок 4, 5, 6: Результаты и интерфейс предсказания ---
if st.session_state.step == 2:
    st.header("3. Результаты обучения")
    metrics = st.session_state.metrics
    c1, c2 = st.columns(2)
    c1.metric("Средняя ошибка (MAE)", f"{metrics['MAE']:.2f} шт.")
    c2.metric("Точность модели (R²)", f"{metrics['R2']:.2%}")

    # --- 7. Аналитика: Важность признаков ---
    feature_importance_df = pd.DataFrame({
        'feature': st.session_state.features,
        'importance': st.session_state.model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h', title='Важность признаков для модели')
    st.plotly_chart(fig, use_container_width=True)

    st.header("4. Ввод новой модели для рекомендации")
    with st.form("prediction_form"):
        st.subheader("🆕 Введите характеристики новой модели")
        new_product_data = {}
        c1, c2 = st.columns(2)
        
        with c1:
            new_product_data['Price'] = st.number_input("💰 Цена модели:", min_value=0.0, step=100.0)
        
        # Динамическое создание полей ввода
        manual_features_to_input = list(st.session_state.feature_config['manual_features'].keys())
        for i, feature in enumerate(manual_features_to_input):
            with c1 if i % 2 == 0 else c2:
                new_product_data[feature] = st.text_input(f"🔹 {feature}:")
                
        # Если используется описание, добавляем поля для авто-извлеченных признаков
        if st.session_state.feature_config['describe_col'] != "Не использовать":
            st.info("Так как модель училась на авто-признаках из описания, введите их вручную:")
            with c2:
                new_product_data['brand_extracted'] = st.text_input("🏷️ Бренд (из описания):")
                new_product_data['material_extracted'] = st.text_input("🔧 Материал (из описания):")
                new_product_data['shape_extracted'] = st.text_input("🕶️ Форма (из описания):")
        
        predict_button = st.form_submit_button("🎯 ПОДОБРАТЬ МАГАЗИНЫ", type="primary")

    if predict_button:
        # --- 6. Система рекомендаций ---
        model = st.session_state.model
        features = st.session_state.features
        all_stores = st.session_state.df_agg['Magazin'].unique()
        
        # Создаем DataFrame для предсказания
        prediction_df = pd.DataFrame(columns=features)
        for store in all_stores:
            row = new_product_data.copy()
            row['Magazin'] = store
            prediction_df.loc[len(prediction_df)] = row

        # Преобразуем типы как в обучении
        for col in prediction_df.columns:
            if col in model.get_cat_feature_indices():
                prediction_df[col] = prediction_df[col].astype(str)
            else:
                 prediction_df[col] = pd.to_numeric(prediction_df[col], errors='coerce').fillna(0)

        predictions = model.predict(prediction_df[features])
        prediction_df['prediction'] = np.maximum(0, predictions)
        
        max_pred = prediction_df['prediction'].max()
        prediction_df['compatibility'] = (prediction_df['prediction'] / max_pred * 100) if max_pred > 0 else 0
        
        results = prediction_df.sort_values('prediction', ascending=False)
        
        st.subheader("🏆 Рекомендуемые магазины")
        for i, row in results.head(5).iterrows():
            with st.expander(f"**#{i+1} {row['Magazin']}** - Прогноз: **{row['prediction']:.0f} шт/мес**"):
                c1, c2 = st.columns(2)
                c1.metric("Прогноз продаж", f"{row['prediction']:.0f} шт")
                c2.metric("Индекс совместимости", f"{row['compatibility']:.0f}%")
        
        st.subheader("❌ Менее подходящие магазины")
        for i, row in results.tail(3).iterrows():
             st.markdown(f"- **{row['Magazin']}**: Прогноз ~{row['prediction']:.0f} шт/мес. Возможно, не лучший выбор.")
