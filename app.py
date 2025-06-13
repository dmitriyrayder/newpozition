import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna
import plotly.express as px
import re
import traceback

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
    if not columns:
        return 0
    
    for keyword in keywords:
        for i, col in enumerate(columns):
            if keyword.lower() in str(col).lower():
                return i
    return min(default_index, len(columns) - 1) if columns else 0

def extract_features_from_description(descriptions_series):
    """Упрощенное извлечение признаков из текстового описания."""
    if descriptions_series.empty:
        return pd.DataFrame()
    
    features = pd.DataFrame(index=descriptions_series.index)
    descriptions_clean = descriptions_series.fillna('').astype(str).str.lower()
    
    extraction_map = {
        'brand_extracted': ['ray-ban', 'oakley', 'gucci', 'prada', 'polaroid'],
        'material_extracted': ['металл', 'пластик', 'дерево', 'комбинированный'],
        'shape_extracted': ['авиатор', 'вайфарер', 'круглые', 'кошачий глаз']
    }
    
    for feature_name, keywords in extraction_map.items():
        results = []
        for desc in descriptions_clean:
            found = 'не определен'
            for keyword in keywords:
                if keyword in desc:
                    found = keyword
                    break
            results.append(found)
        features[feature_name] = results
    
    return features

@st.cache_data
def process_data_and_train(_df, column_map, feature_config):
    """Основная функция: обрабатывает данные, извлекает признаки, обучает модель."""
    try:
        df = _df.copy()
        
        missing_columns = [f"`{v}` (для `{k}`)" for k, v in column_map.items() if v and v not in df.columns]
        if missing_columns:
            return None, None, None, None, f"Отсутствуют колонки в данных: {', '.join(missing_columns)}"

        all_features_df = pd.DataFrame(index=df.index)
        
        if feature_config['describe_col'] != "Не использовать":
            user_selected_describe_col = feature_config['describe_col']
            if user_selected_describe_col in df.columns and not df[user_selected_describe_col].empty:
                try:
                    extracted = extract_features_from_description(df[user_selected_describe_col])
                    if not extracted.empty:
                        all_features_df = pd.concat([all_features_df, extracted], axis=1)
                except Exception as e:
                    st.warning(f"Не удалось извлечь признаки из описания: {e}")

        for feature, source_col in feature_config['manual_features'].items():
            if source_col and source_col in df.columns and not df[source_col].empty:
                all_features_df[feature] = df[source_col].fillna('не определен').astype(str)

        df.rename(columns={v: k for k, v in column_map.items() if v}, inplace=True)
        required_cols = ['date', 'Art', 'Magazin', 'Qty', 'Price']
        if any(col not in df.columns for col in required_cols):
            return None, None, None, None, f"Отсутствуют обязательные колонки после переименования: `date`, `Art`, `Magazin`, `Qty`, `Price`"

        initial_len = len(df)
        df.dropna(subset=required_cols, inplace=True)
        if len(df) == 0:
            return None, None, None, None, "Все строки были удалены, так как в них отсутствовали значения в обязательных полях."

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        if len(df) == 0:
            return None, None, None, None, "Все строки удалены из-за некорректного формата даты."

        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        df = df[(df['Price'] > 0) & (df['Qty'] > 0)]
        final_len = len(df)

        if final_len == 0:
            return None, None, None, None, "Не найдено строк с положительными значениями цены и количества."
        
        data_loss_percent = (initial_len - final_len) / initial_len * 100
        st.info(f"📊 **Статистика очистки:** Исходных строк: {initial_len:,}. Строк для обучения: {final_len:,}. Удалено: {data_loss_percent:.1f}%")
        if data_loss_percent > 50:
            st.warning("⚠️ Удалено более 50% данных. Проверьте качество исходного файла.")

        if not all_features_df.empty:
            df = pd.concat([df, all_features_df.reindex(df.index)], axis=1)
        
        df = df.sort_values(by=['Art', 'Magazin', 'date'])
        first_sale_dates = df.groupby(['Art', 'Magazin'])['date'].first().reset_index(name='first_sale_date')
        df_merged = pd.merge(df, first_sale_dates, on=['Art', 'Magazin'])
        df_30_days = df_merged[df_merged['date'] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))].copy()

        agg_logic = {'Qty': 'sum', 'Price': 'mean'}
        feature_cols = [col for col in all_features_df.columns if col in df_30_days.columns]
        for col in feature_cols:
            agg_logic[col] = 'first'
        
        df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
        df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)

        if len(df_agg) < 10:
            return None, None, None, None, f"Слишком мало агрегированных данных для обучения: {len(df_agg)} записей. Нужно минимум 10."

        target = 'Qty_30_days'
        cat_features_to_use = ['Magazin'] + feature_cols
        features_to_use = ['Price'] + cat_features_to_use
        
        X = df_agg[features_to_use].copy()
        y = df_agg[target]
        for col in cat_features_to_use:
            X[col] = X[col].fillna('не определен').astype('category')

        if len(X) < 5:
            return None, None, None, None, f"Слишком мало данных для разделения на выборки: {len(X)}."

        test_size = min(0.2, max(0.1, 5.0 / len(X)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        def objective(trial):
            params = {
                'iterations': 100,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 3, 6),
                'verbose': 0, 'random_seed': 42
            }
            model = CatBoostRegressor(**params, cat_features=cat_features_to_use)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, use_best_model=True)
            return mean_absolute_error(y_test, model.predict(X_test))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        
        final_model = CatBoostRegressor(**study.best_params, iterations=200, verbose=0, random_seed=42, cat_features=cat_features_to_use)
        final_model.fit(X, y)
        
        y_pred = final_model.predict(X_test)
        metrics = {'MAE': mean_absolute_error(y_test, y_pred), 'R2': max(0, r2_score(y_test, y_pred))}
        
        unique_values_for_prediction = {col: X[col].unique().tolist() for col in cat_features_to_use}

        return final_model, features_to_use, metrics, unique_values_for_prediction, None
        
    except Exception as e:
        error_details = traceback.format_exc()
        return None, None, None, None, f"Критическая ошибка обработки данных: {str(e)}\n\nДетали:\n{error_details}"

# ==============================================================================
# ОСНОВНОЙ КОД ПРИЛОЖЕНИЯ
# ==============================================================================

st.title("💖 Модный Советник по Продажам")

if 'step' not in st.session_state:
    st.session_state.step = 1

with st.sidebar:
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл Excel или CSV", type=["csv", "xlsx", "xls"], key="file_uploader")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp1251')
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            if df.empty:
                st.error("Загруженный файл пустой!")
                st.stop()
            
            # ===== ИСПРАВЛЕНИЕ ЗДЕСЬ =====
            # Старый код: df.columns = df.columns.astype(str).str.strip()
            # Новый, надежный код:
            df.columns = [str(col).strip() for col in df.columns]
            # ==============================

            st.session_state.df_raw = df
            st.session_state.step = 1
            
            st.success(f"📊 Файл '{uploaded_file.name}' успешно загружен!")
            col1, col2 = st.columns(2)
            col1.metric("Строк", f"{len(df):,}")
            col2.metric("Колонок", f"{len(df.columns):,}")
            
            with st.expander("👀 Превью и статистика данных"):
                st.dataframe(df.head(10))
                st.write("**Статистика по заполнению колонок:**")
                col_stats = pd.DataFrame({
                    'Заполнено, %': (df.count() / len(df) * 100).round(1),
                    'Пустых значений': df.isnull().sum(),
                })
                st.dataframe(col_stats, use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

if 'df_raw' in st.session_state:
    df_raw = st.session_state.df_raw
    available_columns = [""] + df_raw.columns.tolist()

    st.header("2. Настройка датасета")
    with st.form("column_mapping_form"):
        st.subheader("🎯 Сопоставление основных колонок")
        c1, c2 = st.columns(2)
        with c1:
            col_magazin = st.selectbox("Магазин:", available_columns, index=auto_detect_column(available_columns, ['magazin', 'магазин']))
            col_art = st.selectbox("Артикул:", available_columns, index=auto_detect_column(available_columns, ['art', 'артикул'], 1))
            col_qty = st.selectbox("Количество:", available_columns, index=auto_detect_column(available_columns, ['qty', 'количество'], 2))
        with c2:
            col_date = st.selectbox("Дата продажи:", available_columns, index=auto_detect_column(available_columns, ['datasales', 'дата'], 3))
            col_price = st.selectbox("Цена:", available_columns, index=auto_detect_column(available_columns, ['price', 'цена'], 4))
            col_describe = st.selectbox("Описание (для авто-признаков):", available_columns + ["Не использовать"], index=auto_detect_column(available_columns, ['describe', 'описание'], 5))

        st.subheader("✨ Дополнительные признаки товара")
        other_feature_cols = st.multiselect(
            "Выберите колонки с характеристиками (Бренд, Цвет, Пол и т.д.):",
            [c for c in df_raw.columns if c not in [col_magazin, col_art, col_qty, col_date, col_price, col_describe] and c != ""]
        )
        
        submitted = st.form_submit_button("✅ Подтвердить и обучить модель", type="primary", use_container_width=True)

    if submitted:
        if not all([col_magazin, col_art, col_qty, col_date, col_price]):
            st.error("Пожалуйста, выберите все обязательные колонки!")
        else:
            column_map = {'Magazin': col_magazin, 'Art': col_art, 'date': col_date, 'Qty': col_qty, 'Price': col_price}
            feature_config = {'describe_col': col_describe, 'manual_features': {col: col for col in other_feature_cols}}
            
            with st.spinner("Магия в процессе... Обрабатываю данные и обучаю модель..."):
                model, features, metrics, unique_vals, error_msg = process_data_and_train(df_raw, column_map, feature_config)

            if error_msg:
                st.error(f"**Произошла ошибка:**\n\n{error_msg}")
                st.session_state.step = 1
            else:
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.metrics = metrics
                st.session_state.unique_values_for_prediction = unique_vals
                st.session_state.feature_config = feature_config
                st.session_state.step = 2
                st.success("Модель успешно обучена! Результаты и форма для прогноза ниже. 👇")
                st.rerun()

if st.session_state.step == 2:
    st.header("3. Результаты обучения модели")
    metrics = st.session_state.metrics
    c1, c2 = st.columns(2)
    c1.metric("Средняя ошибка прогноза (MAE)", f"{metrics['MAE']:.2f} шт.", help="В среднем модель ошибается на это количество единиц товара.")
    c2.metric("Точность модели (R²)", f"{metrics['R2']:.1%}", help="Насколько хорошо модель объясняет данные (чем ближе к 100%, тем лучше).")

    try:
        feature_importance_df = pd.DataFrame({
            'Признак': st.session_state.features,
            'Важность': st.session_state.model.get_feature_importance()
        }).sort_values('Важность', ascending=False)
        fig = px.bar(feature_importance_df, x='Важность', y='Признак', orientation='h', title='Наиболее важные признаки для прогноза продаж')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Не удалось отобразить важность признаков: {e}")

    st.header("4. Сделать прогноз")
    st.info("Введите данные нового товара, чтобы спрогнозировать его продажи за первые 30 дней.")
    
    unique_vals = st.session_state.unique_values_for_prediction
    
    with st.form("prediction_form"):
        prediction_data = {}
        cols = st.columns(2)
        
        col_idx = 0
        for feature in st.session_state.features:
            current_col = cols[col_idx % 2]
            with current_col:
                if feature == 'Price':
                    prediction_data[feature] = st.number_input("Цена товара", min_value=0.0, step=100.0, value=1000.0)
                elif feature in unique_vals:
                    options = sorted(unique_vals[feature])
                    prediction_data[feature] = st.selectbox(f"Признак: {feature}", options=options, index=0)

        predict_button = st.form_submit_button("🔮 Спрогнозировать продажи", type="primary", use_container_width=True)

    if predict_button:
        try:
            input_df = pd.DataFrame([prediction_data])
            for col in input_df.columns:
                if col in st.session_state.model.get_cat_feature_indices():
                    input_df[col] = input_df[col].astype('category')

            prediction = st.session_state.model.predict(input_df)
            predicted_qty = int(round(prediction[0]))
            
            st.success(f"### 📈 Прогноз продаж: **~{predicted_qty} шт.**")
            st.caption("Это прогнозируемое количество товара, которое будет продано в указанном магазине за первые 30 дней.")

        except Exception as e:
            st.error(f"Ошибка при прогнозировании: {e}")
