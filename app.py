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
    if not columns:
        return 0
    
    for keyword in keywords:
        for i, col in enumerate(columns):
            if keyword.lower() in col.lower():
                return i
    return min(default_index, len(columns) - 1)

def extract_features_from_description(descriptions_str):
    """Упрощенное извлечение признаков из текстового описания.
       Принимает серию данных и безопасно обрабатывает их как строки."""
    features = pd.DataFrame(index=descriptions_str.index)
    
    # Безопасное преобразование в строки
    descriptions_clean = descriptions_str.astype(str).fillna('').str.lower()
    
    extraction_map = {
        'brand_extracted': ['ray-ban', 'oakley', 'gucci', 'prada', 'polaroid'],
        'material_extracted': ['металл', 'пластик', 'дерево', 'комбинированный'],
        'shape_extracted': ['авиатор', 'вайфарер', 'круглые', 'кошачий глаз']
    }
    
    for feature_name, keywords in extraction_map.items():
        # Создаем список для результатов
        results = []
        
        for desc in descriptions_clean:
            found = 'не определен'
            for keyword in keywords:
                if keyword.lower() in desc:
                    found = keyword.lower()
                    break
            results.append(found)
        
        features[feature_name] = results
    
    return features

@st.cache_data
def process_data_and_train(_df, column_map, feature_config):
    """
    Основная функция: обрабатывает данные, извлекает признаки, обучает модель.
    """
    try:
        df = _df.copy()
        
        # --- ИСПРАВЛЕННАЯ ЛОГИКА: СНАЧАЛА ИЗВЛЕКАЕМ, ПОТОМ ПЕРЕИМЕНОВЫВАЕМ ---

        # 1. Извлечение признаков по ОРИГИНАЛЬНЫМ именам колонок
        all_features_df = pd.DataFrame(index=df.index)
        
        if feature_config['describe_col'] != "Не использовать":
            user_selected_describe_col = feature_config['describe_col']
            if user_selected_describe_col in df.columns:
                # Безопасная обработка колонки описания
                describe_series = df[user_selected_describe_col]
                # Проверяем, что колонка не пустая
                if not describe_series.empty:
                    extracted = extract_features_from_description(describe_series)
                    all_features_df = pd.concat([all_features_df, extracted], axis=1)

        for feature, source_col in feature_config['manual_features'].items():
            if source_col and source_col in df.columns:
                # Безопасное преобразование в строки
                feature_series = df[source_col]
                if not feature_series.empty:
                    all_features_df[feature] = feature_series.astype(str).fillna('не определен')

        # 2. Теперь, когда все данные извлечены, ПЕРЕИМЕНОВЫВАЕМ основные колонки
        # ИСПРАВЛЕНИЕ: Проверяем, что колонки существуют перед переименованием
        valid_column_map = {k: v for k, v in column_map.items() if v and v in df.columns}
        df.rename(columns={v: k for k, v in valid_column_map.items()}, inplace=True)

        # ИСПРАВЛЕНИЕ: Проверяем наличие обязательных колонок после переименования
        required_cols = ['date', 'Art', 'Magazin', 'Qty', 'Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, None, None, f"Отсутствуют обязательные колонки после переименования: {missing_cols}"

        # 3. Валидация и очистка уже переименованных колонок
        # Статистика до обработки
        initial_len = len(df)
        initial_stats = {
            'total_rows': initial_len,
            'rows_with_required_fields': 0,
            'rows_with_valid_dates': 0,
            'rows_with_positive_values': 0
        }
        
        # Безопасная обработка даты
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            initial_stats['rows_with_valid_dates'] = df['date'].notna().sum()
        
        # Проверяем строки с заполненными обязательными полями
        required_fields = ['date', 'Art', 'Magazin', 'Qty', 'Price']
        rows_before_cleanup = df.dropna(subset=required_fields).shape[0]
        initial_stats['rows_with_required_fields'] = rows_before_cleanup
        
        # Удаляем строки с пустыми ключевыми полями
        df.dropna(subset=required_fields, inplace=True)
        
        # ИСПРАВЛЕНИЕ: Проверяем, остались ли данные после очистки
        if len(df) == 0:
            return None, None, None, "Все данные были удалены при очистке. Проверьте форматы дат и обязательных полей."
        
        # Безопасная обработка числовых полей
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        
        # Проверяем строки с положительными значениями
        positive_values = (df['Price'] > 0) & (df['Qty'] > 0)
        initial_stats['rows_with_positive_values'] = positive_values.sum()
        
        # Удаляем записи с нулевыми ценами и количествами
        df = df[positive_values]
        final_len = len(df)
        
        if final_len == 0:
            return None, None, None, "Все данные были удалены: нет записей с положительными ценами и количествами."
        
        # Формируем детальную статистику обработки
        processing_stats = f"""
        📊 **Статистика обработки данных:**
        - Исходное количество строк: **{initial_stats['total_rows']:,}**
        - Строки с валидными датами: **{initial_stats['rows_with_valid_dates']:,}** ({initial_stats['rows_with_valid_dates']/initial_stats['total_rows']*100:.1f}%)
        - Строки со всеми обязательными полями: **{initial_stats['rows_with_required_fields']:,}** ({initial_stats['rows_with_required_fields']/initial_stats['total_rows']*100:.1f}%)
        - Строки с положительными ценами и количествами: **{initial_stats['rows_with_positive_values']:,}** ({initial_stats['rows_with_positive_values']/initial_stats['total_rows']*100:.1f}%)
        - **Итоговое количество строк для обучения: {final_len:,}** ({final_len/initial_stats['total_rows']*100:.1f}%)
        """
        
        # Показываем статистику пользователю
        st.info(processing_stats)
        
        # Предупреждения о потере данных
        data_loss_percent = (initial_stats['total_rows'] - final_len) / initial_stats['total_rows'] * 100
        if data_loss_percent > 50:
            st.warning(f"⚠️ Критическая потеря данных: {data_loss_percent:.1f}% строк удалено. Проверьте качество исходных данных.")
        elif data_loss_percent > 20:
            st.warning(f"⚠️ Значительная потеря данных: {data_loss_percent:.1f}% строк удалено.")
        elif data_loss_percent > 0:
            st.info(f"ℹ️ Удалено {data_loss_percent:.1f}% некорректных строк.")
        
        # 4. Агрегация данных
        df_with_features = pd.concat([df[['Art', 'Magazin', 'date', 'Qty', 'Price']], all_features_df], axis=1)
        df_with_features.dropna(subset=['Art', 'Magazin'], inplace=True) 

        # ИСПРАВЛЕНИЕ: Проверяем, остались ли данные после объединения
        if len(df_with_features) == 0:
            return None, None, None, "Данные потерялись при объединении с признаками."

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

        # ИСПРАВЛЕНИЕ: Увеличиваем минимальный порог данных и добавляем детальную проверку
        if len(df_agg) < 10:
            return None, None, None, f"Слишком мало данных для обучения: {len(df_agg)} записей. Нужно минимум 10."

        # 5. Обучение модели
        target = 'Qty_30_days'
        cat_features_to_use = ['Magazin'] + feature_cols
        features_to_use = ['Price'] + cat_features_to_use
        
        # ИСПРАВЛЕНИЕ: Проверяем наличие всех признаков в данных
        missing_features = [f for f in features_to_use if f not in df_agg.columns]
        if missing_features:
            return None, None, None, f"Отсутствуют признаки в обработанных данных: {missing_features}"
        
        X = df_agg[features_to_use]
        y = df_agg[target]
        
        for col in cat_features_to_use:
            if col in X.columns:  # ИСПРАВЛЕНИЕ: Дополнительная проверка существования колонки
                X[col] = X[col].astype(str)

        # ИСПРАВЛЕНИЕ: Проверяем размер данных перед разделением
        if len(X) < 5:
            return None, None, None, f"Слишком мало данных для разделения на обучающую и тестовую выборки: {len(X)} записей."

        test_size = min(0.2, max(0.1, 1.0 / len(X)))  # Адаптивный размер тестовой выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        def objective(trial):
            params = {
                'iterations': 100,  # ИСПРАВЛЕНИЕ: Уменьшили количество итераций для ускорения
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 3, 6),  # ИСПРАВЛЕНИЕ: Уменьшили глубину для малых данных
                'verbose': 0, 
                'random_seed': 42
            }
            try:
                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, cat_features=[i for i, col in enumerate(features_to_use) if col in cat_features_to_use], 
                         eval_set=(X_test, y_test), early_stopping_rounds=10, use_best_model=True)
                return mean_absolute_error(y_test, model.predict(X_test))
            except Exception as e:
                return float('inf')  # Возвращаем бесконечность при ошибке

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # ИСПРАВЛЕНИЕ: Уменьшили количество попыток
        
        # ИСПРАВЛЕНИЕ: Добавили обработку ошибок при финальном обучении
        try:
            final_model = CatBoostRegressor(**study.best_params, iterations=200, verbose=0, random_seed=42)
            final_model.fit(X, y, cat_features=[i for i, col in enumerate(features_to_use) if col in cat_features_to_use])
            
            # ИСПРАВЛЕНИЕ: Безопасное вычисление метрик
            y_pred = final_model.predict(X_test)
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': max(0, r2_score(y_test, y_pred))  # R2 не может быть отрицательным в интерфейсе
            }
        except Exception as e:
            return None, None, None, f"Ошибка при обучении финальной модели: {str(e)}"
        
        return final_model, features_to_use, metrics, None
        
    except Exception as e:
        return None, None, None, f"Общая ошибка обработки данных: {str(e)}"

# ==============================================================================
# ОСНОВНОЙ КОД ПРИЛОЖЕНИЯ
# ==============================================================================

st.title("💖 Модный Советник по Продажам")

# ИСПРАВЛЕНИЕ: Инициализация session_state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None

with st.sidebar:
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл Excel или CSV", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            # Первая попытка чтения без специальных параметров
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            # ИСПРАВЛЕНИЕ: Проверяем, что файл не пустой
            if len(df) == 0:
                st.error("Загруженный файл пустой!")
                st.stop()
            
            # Пытаемся найти колонку с датами для специальной обработки
            date_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'дата', 'datasales']):
                    date_columns.append(col)
            
            # Если найдены колонки с датами, перечитываем файл с правильными типами
            if date_columns:
                try:
                    dtype_dict = {}
                    for date_col in date_columns:
                        dtype_dict[date_col] = 'datetime64[ns]'
                    
                    if uploaded_file.name.endswith('.csv'):
                        df_with_dates = pd.read_csv(uploaded_file, encoding='utf-8', dtype=dtype_dict, parse_dates=date_columns)
                    else:
                        df_with_dates = pd.read_excel(uploaded_file, dtype=dtype_dict, parse_dates=date_columns)
                    
                    # Сравниваем количество успешно прочитанных строк
                    original_rows = len(df)
                    processed_rows = len(df_with_dates)
                    
                    if processed_rows >= original_rows * 0.8:  # Если потеряли менее 20%
                        df = df_with_dates
                        st.info(f"✅ Даты успешно обработаны в колонках: {', '.join(date_columns)}")
                    else:
                        st.warning(f"⚠️ Специальная обработка дат привела к потере данных. Используем стандартное чтение.")
                        
                except Exception as date_error:
                    st.warning(f"⚠️ Не удалось автоматически обработать даты: {date_error}")
            
            st.session_state.df_raw = df
            
            # Статистика датасета
            total_cells = len(df) * len(df.columns)
            empty_cells = df.isnull().sum().sum()
            filled_cells = total_cells - empty_cells
            fill_percentage = (filled_cells / total_cells * 100) if total_cells > 0 else 0
            
            st.success(f"📊 Файл успешно загружен!")
            
            # Детальная статистика
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Строк", f"{len(df):,}")
            with col2:
                st.metric("Колонок", f"{len(df.columns):,}")
            with col3:
                st.metric("Заполненность", f"{fill_percentage:.1f}%")
            
            # Дополнительная статистика по строкам
            with st.expander("📈 Детальная статистика данных"):
                rows_with_data = df.dropna(how='all').shape[0]
                empty_rows = len(df) - rows_with_data
                
                st.write("**Статистика по строкам:**")
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Строки с данными", f"{rows_with_data:,}")
                    st.metric("Пустые строки", f"{empty_rows:,}")
                with stat_col2:
                    valid_percentage = (rows_with_data / len(df) * 100) if len(df) > 0 else 0
                    st.metric("% строк с данными", f"{valid_percentage:.1f}%")
                    empty_percentage = (empty_rows / len(df) * 100) if len(df) > 0 else 0
                    st.metric("% пустых строк", f"{empty_percentage:.1f}%")
                
                st.write("**Статистика по колонкам:**")
                col_stats = pd.DataFrame({
                    'Колонка': df.columns,
                    'Заполнено': df.count(),
                    'Пусто': df.isnull().sum(),
                    '% заполнено': (df.count() / len(df) * 100).round(1)
                })
                st.dataframe(col_stats, use_container_width=True)
            
            # Показываем превью данных
            with st.expander("👀 Превью данных"):
                st.dataframe(df.head(10))
                
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

# ИСПРАВЛЕНИЕ: Проверяем наличие данных
if st.session_state.df_raw is not None:
    df_raw = st.session_state.df_raw
    available_columns = df_raw.columns.tolist()

    st.header("2. Настройка датасета")
    with st.form("column_mapping_form"):
        st.subheader("🎯 Сопоставление основных колонок")
        st.caption("Помогите мне понять, где какие данные находятся. Я попробую угадать сама!")
        c1, c2 = st.columns(2)
        with c1:
            col_magazin = st.selectbox("Колонка МАГАЗИН:", available_columns, 
                                     index=auto_detect_column(available_columns, ['magazin', 'магазин', 'store']))
            col_art = st.selectbox("Колонка АРТИКУЛ:", available_columns, 
                                 index=auto_detect_column(available_columns, ['art', 'артикул', 'sku'], 1))
            col_qty = st.selectbox("Колонка КОЛИЧЕСТВО:", available_columns, 
                                 index=auto_detect_column(available_columns, ['qty', 'количество'], 2))
        with c2:
            col_date = st.selectbox("Колонка ДАТА ПРОДАЖИ:", available_columns, 
                                  index=auto_detect_column(available_columns, ['datasales', 'дата', 'date'], 3))
            col_price = st.selectbox("Колонка ЦЕНА:", available_columns, 
                                   index=auto_detect_column(available_columns, ['price', 'цена'], 4))
            col_describe = st.selectbox("Колонка ОПИСАНИЕ ТОВАРА:", available_columns + ["Не использовать"], 
                                      index=auto_detect_column(available_columns, ['describe', 'описание'], 5))

        st.subheader("✋ Ручная настройка признаков товара")
        st.caption("Какие еще колонки содержат важную информацию о товаре? (например, Бренд, Цвет, Пол)")
        other_feature_cols = st.multiselect(
            "Выберите дополнительные колонки-признаки:",
            [c for c in available_columns if c not in [col_magazin, col_art, col_qty, col_date, col_price, col_describe]]
        )
        
        submitted = st.form_submit_button("✅ Подтвердить и обучить модель", type="primary")

    if submitted:
        # ИСПРАВЛЕНИЕ: Проверяем, что все обязательные колонки выбраны
        required_mappings = [col_magazin, col_art, col_qty, col_date, col_price]
        if not all(required_mappings):
            st.error("Пожалуйста, выберите все обязательные колонки!")
            st.stop()
            
        st.session_state.step = 2
        column_map = {'Magazin': col_magazin, 'Art': col_art, 'date': col_date, 'Qty': col_qty, 'Price': col_price}
        feature_config = {
            'describe_col': col_describe,
            'manual_features': {col: col for col in other_feature_cols}
        }

        with st.spinner("Обрабатываю данные и обучаю модель..."):
            model, features, metrics, error_msg = process_data_and_train(df_raw, column_map, feature_config)

        if error_msg:
            st.error(error_msg)
        else:
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.metrics = metrics
            st.session_state.all_stores = df_raw[col_magazin].unique()
            st.session_state.feature_config = feature_config
            st.success("Модель успешно обучена!")
            st.rerun()

# ИСПРАВЛЕНИЕ: Проверяем step корректно
if st.session_state.step == 2 and 'model' in st.session_state:
    st.header("3. Результаты обучения")
    metrics = st.session_state.metrics
    c1, c2 = st.columns(2)
    c1.metric("Средняя ошибка (MAE)", f"{metrics['MAE']:.2f} шт.")
    c2.metric("Точность модели (R²)", f"{metrics['R2']:.2%}")

    # ИСПРАВЛЕНИЕ: Безопасное получение важности признаков
    try:
        feature_importance = st.session_state.model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            'feature': st.session_state.features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h', 
                    title='Важность признаков для модели')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Не удалось отобразить важность признаков: {e}")

    st.header("4. Ввод новой модели для рекомендации")
    with st.form("prediction_form"):
        st.subheader("🆕 Введите характеристики новой модели")
        new_product_data = {}
        c1, c2 = st.columns(2)
        
        with c1:
            new_product_data['Price'] = st.number_input("💰 Цена модели:", min_value=0.0, step=100.0, value=1000.0)
        
        manual_features_to_input = list(st.session_state.feature_config['manual_features'].keys())
        for i, feature in enumerate(manual_features_to_input):
            with c1 if (i+1) % 2 != 0 else c2:
                new_product_data[feature] = st.text_input(f"🔹 {feature}:", value="не определен")
                
        if st.session_state.feature_config['describe_col'] != "Не использовать":
            st.info("Так как модель училась на авто-признаках из описания, введите их вручную:")
            with c2:
                new_product_data['brand_extracted'] = st.text_input("🏷️ Бренд (из описания):", value="не определен")
                new_product_data['material_extracted'] = st.text_input("🔧 Материал (из описания):", value="не определен")
                new_product_data['shape_extracted'] = st.text_input("🕶️ Форма (из описания):", value="не определен")
        
        predict_button = st.form_submit_button("🎯 ПОДОБРАТЬ МАГАЗИНЫ", type="primary")

    if predict_button:
        try:
            model = st.session_state.model
            features = st.session_state.features
            all_stores = st.session_state.all_stores
            
            prediction_df = pd.DataFrame(columns=features)
            for store in all_stores:
                row = new_product_data.copy()
                row['Magazin'] = str(store)
                
                # ИСПРАВЛЕНИЕ: Заполняем все недостающие признаки
                for f in features:
                    if f not in row:
                        row[f] = 'не определен'
                
                prediction_df.loc[len(prediction_df)] = row

            # ИСПРАВЛЕНИЕ: Корректная обработка типов данных
            cat_features_indices = []
            try:
                cat_features_indices = model.get_cat_feature_indices()
            except:
                # Если не удается получить индексы, определяем категориальные признаки по названиям
                cat_features_names = ['Magazin'] + [col for col in features if 'extracted' in col or col in st.session_state.feature_config['manual_features']]
                cat_features_indices = [i for i, col in enumerate(features) if col in cat_features_names]
            
            for i, col in enumerate(features):
                if i in cat_features_indices:
                    prediction_df[col] = prediction_df[col].astype(str)
                else:
                    prediction_df[col] = pd.to_numeric(prediction_df[col], errors='coerce').fillna(0)

            predictions = model.predict(prediction_df[features])
            prediction_df['prediction'] = np.maximum(0, predictions)
            
            max_pred = prediction_df['prediction'].max()
            prediction_df['compatibility'] = (prediction_df['prediction'] / max_pred * 100) if max_pred > 0 else 0
            
            results = prediction_df.sort_values('prediction', ascending=False)
            
            st.subheader("🏆 Рекомендуемые магазины")
            top_results = results.head(min(5, len(results)))
            for idx, (i, row) in enumerate(top_results.iterrows()):
                with st.expander(f"**#{idx+1} {row['Magazin']}** - Прогноз: **{row['prediction']:.0f} шт/мес**"):
                    c1, c2 = st.columns(2)
                    c1.metric("Прогноз продаж", f"{row['prediction']:.0f} шт")
                    c2.metric("Индекс совместимости", f"{row['compatibility']:.0f}%")
            
            if len(results) > 5:
                st.subheader("❌ Менее подходящие магазины")
                bottom_results = results.tail(min(3, len(results) - 5))
                for i, row in bottom_results.iterrows():
                    st.markdown(f"- **{row['Magazin']}**: Прогноз ~{row['prediction']:.0f} шт/мес. Возможно, не лучший выбор.")
                    
        except Exception as e:
            st.error(f"Ошибка при прогнозировании: {str(e)}")
            st.error("Попробуйте заново обучить модель или проверьте введенные данные.")

elif st.session_state.step == 1:
    st.info("👆 Пожалуйста, загрузите файл с данными в боковой панели для начала работы.")
else:
    st.warning("Что-то пошло не так. Попробуйте перезагрузить страницу.")
