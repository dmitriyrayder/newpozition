import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna
import logging
import warnings
import plotly.express as px
from datetime import datetime, timedelta

# Настройка
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Модный Советник", layout="wide")

# Стили
st.markdown("""
<style>
.main { background-color: #fce4ec; }
h1 { font-family: 'Comic Sans MS', cursive, sans-serif; color: #e91e63; text-align: center; }
.stButton>button {
    color: white; background: linear-gradient(to right, #f06292, #e91e63); 
    border-radius: 25px; border: 2px solid #ad1457; padding: 12px 28px; font-weight: bold;
}
.metric-card { padding: 10px; border-radius: 10px; background-color: #fff1f8; border: 1px solid #f8bbd0; }
</style>
""", unsafe_allow_html=True)

st.title("💖 Модный Советник по Продажам")

# Функции
@st.cache_data
def load_data(file):
    """Загрузка данных с обработкой ошибок"""
    try:
        if file.size > 50 * 1024 * 1024:
            st.error("Файл слишком большой! Максимум 50MB")
            return None
            
        if file.name.endswith('.csv'):
            for encoding in ['utf-8', 'cp1251', 'latin1']:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    logger.info(f"CSV загружен с {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("Не удалось определить кодировку")
            return None
        else:
            df = pd.read_excel(file)
            logger.info("Excel файл загружен")
            return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

def parse_dates_robust(df, date_col):
    """Принудительная обработка дат"""
    df = df.copy()
    original_count = len(df)
    
    # Принудительное преобразование
    if date_col in df.columns:
        # Сначала стандартное преобразование
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Если много NaT, пробуем различные форматы
        if df[date_col].isna().sum() > len(df) * 0.1:
            st.warning("⚠️ Проблемы с датами, применяем принудительное преобразование")
            
            # Попытка с dtype
            try:
                df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
            except:
                # Форматы для попытки
                formats = ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
                for fmt in formats:
                    mask = df[date_col].isna()
                    if mask.sum() == 0:
                        break
                    try:
                        df.loc[mask, date_col] = pd.to_datetime(
                            df.loc[mask, date_col].astype(str), format=fmt, errors='coerce'
                        )
                    except:
                        continue
    
    # Статистика по датам
    bad_dates = df[date_col].isna().sum()
    bad_dates_pct = (bad_dates / original_count) * 100 if original_count > 0 else 0
    
    return df, bad_dates, bad_dates_pct

@st.cache_data
def process_data(_df, art_col, magazin_col, date_col, qty_col, price_col, cat_features):
    """Обработка и агрегация данных"""
    df = _df.copy()
    initial_rows = len(df)
    
    # Переименование колонок
    rename_map = {art_col: 'Art', magazin_col: 'Magazin', date_col: 'date', 
                  qty_col: 'Qty', price_col: 'Price'}
    df.rename(columns=rename_map, inplace=True)
    
    # Обработка дат с принудительным преобразованием
    df, bad_dates, bad_dates_pct = parse_dates_robust(df, 'date')
    
    # Очистка данных
    df.dropna(subset=['date', 'Qty', 'Art', 'Magazin', 'Price'], inplace=True)
    df = df[(df['Qty'] > 0) & (df['Price'] > 0)]
    
    # Удаление выбросов (IQR метод)
    for col in ['Qty', 'Price']:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    # Агрегация за первые 30 дней
    df = df.sort_values(['Art', 'Magazin', 'date'])
    first_sales = df.groupby(['Art', 'Magazin'])['date'].first().reset_index()
    first_sales.rename(columns={'date': 'first_date'}, inplace=True)
    
    df = pd.merge(df, first_sales, on=['Art', 'Magazin'])
    df = df[df['date'] <= (df['first_date'] + pd.Timedelta(days=30))]
    
    # Дополнительные признаки
    df['days_since_launch'] = (df['date'] - df['first_date']).dt.days
    df['revenue'] = df['Qty'] * df['Price']
    
    # Агрегация
    agg_dict = {
        'Qty': ['sum', 'mean', 'std'],
        'Price': ['mean', 'std'],
        'revenue': 'sum',
        'days_since_launch': 'max'
    }
    
    for cat_col in cat_features:
        if cat_col in df.columns:
            agg_dict[cat_col] = 'first'
    
    df_agg = df.groupby(['Art', 'Magazin']).agg(agg_dict).reset_index()
    df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns]
    
    # Переименование колонок
    rename_dict = {
        'Qty_sum': 'Qty_30_days', 'Qty_mean': 'Avg_daily_qty', 'Qty_std': 'Qty_volatility',
        'Price_mean': 'Price', 'Price_std': 'Price_volatility',
        'revenue_sum': 'Total_revenue', 'days_since_launch_max': 'Days_in_sale'
    }
    df_agg.rename(columns=rename_dict, inplace=True)
    df_agg['Qty_volatility'] = df_agg['Qty_volatility'].fillna(0)
    df_agg['Price_volatility'] = df_agg['Price_volatility'].fillna(0)
    
    # Статистика обработки
    stats = {
        "initial_rows": initial_rows,
        "final_rows": len(df_agg),
        "bad_dates": bad_dates,
        "bad_dates_pct": bad_dates_pct,
        "removed_rows": initial_rows - len(df_agg),
        "removed_pct": ((initial_rows - len(df_agg)) / initial_rows * 100) if initial_rows > 0 else 0,
        "unique_products": df_agg['Art'].nunique(),
        "unique_stores": df_agg['Magazin'].nunique()
    }
    
    return df_agg, stats

@st.cache_resource
def train_model(_df_agg, cat_features, n_trials=30):
    """Обучение модели с оптимизацией"""
    target = 'Qty_30_days'
    base_features = ['Magazin', 'Price', 'Avg_daily_qty', 'Price_volatility', 'Days_in_sale']
    features = base_features + list(cat_features)
    available_features = [f for f in features if f in _df_agg.columns]
    
    # Подготовка данных
    df_processed = _df_agg[available_features + [target]].copy()
    cat_features_list = ['Magazin'] + [f for f in cat_features if f in df_processed.columns]
    
    for col in cat_features_list:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
    
    X, y = df_processed[available_features], df_processed[target]
    test_size = min(0.25, max(0.1, 20 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Оптимизация параметров
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'verbose': False, 'random_seed': 42
        }
        
        try:
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, cat_features=cat_features_list, 
                     eval_set=(X_test, y_test), early_stopping_rounds=50, 
                     use_best_model=True, verbose=False)
            return mean_absolute_error(y_test, model.predict(X_test))
        except:
            return float('inf')
    
    # Обучение с прогресс-баром
    progress_bar = st.progress(0)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    
    for i in range(n_trials):
        study.optimize(objective, n_trials=1)
        progress_bar.progress((i + 1) / n_trials)
    
    progress_bar.empty()
    
    # Финальная модель
    best_params = study.best_params
    best_params.update({'verbose': False, 'random_seed': 42})
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X, y, cat_features=cat_features_list, verbose=False)
    
    # Метрики
    test_preds = final_model.predict(X_test)
    train_preds = final_model.predict(X_train)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, test_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, test_preds)),
        'R2': r2_score(y_test, test_preds),
        'overfit_ratio': mean_absolute_error(y_train, train_preds) / mean_absolute_error(y_test, test_preds),
        'feature_importance': dict(zip(available_features, final_model.feature_importances_))
    }
    
    return final_model, available_features, metrics

def create_prediction_form(cat_features, df_agg):
    """Форма для прогнозирования"""
    st.subheader("✍️ Характеристики новой модели")
    
    with st.form("prediction_form"):
        data = {}
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Характеристики товара:**")
            for feature in cat_features:
                if feature in df_agg.columns:
                    top_values = df_agg[feature].value_counts().head(3).index.tolist()
                    data[feature] = st.text_input(
                        f"{feature} ✨", 
                        placeholder=f"Например: {top_values[0]}" if top_values else "Введите значение"
                    )
        
        with col2:
            st.markdown("**Параметры продаж:**")
            if 'Price' in df_agg.columns:
                price_stats = df_agg['Price'].describe()
                data['Price'] = st.number_input(
                    "Цена 💰", min_value=1.0, value=float(price_stats['mean']), 
                    step=1.0, format="%.2f"
                )
            
            data['Avg_daily_qty'] = st.number_input(
                "Ожидаемые продажи/день", min_value=0.1, value=1.0, step=0.1
            )
            data['Days_in_sale'] = st.slider("Дни в продаже", 1, 30, 30)
            data['Price_volatility'] = st.number_input("Волатильность цены", 0.0, value=0.0, step=0.1)
        
        return st.form_submit_button("🔮 Создать прогноз!", type="primary"), data

def make_predictions(model, features, product_data, df_agg):
    """Создание прогнозов"""
    # Валидация
    for key, value in product_data.items():
        if key not in ['Price', 'Avg_daily_qty', 'Days_in_sale', 'Price_volatility']:
            if pd.isna(value) or str(value).strip() == "":
                st.error(f"Поле '{key}' не может быть пустым!")
                return pd.DataFrame()
    
    # Создание данных для прогноза
    stores = df_agg['Magazin'].unique()
    predictions_data = []
    
    for store in stores:
        row = product_data.copy()
        row['Magazin'] = store
        predictions_data.append(row)
    
    pred_df = pd.DataFrame(predictions_data)
    available_features = [f for f in features if f in pred_df.columns]
    
    try:
        # Прогнозы
        raw_preds = model.predict(pred_df[available_features])
        pred_df['Прогноз продаж (30 дней, шт.)'] = np.maximum(0, np.round(raw_preds, 0))
        
        # Доверительные интервалы
        pred_std = raw_preds.std()
        pred_df['Мин. прогноз'] = np.maximum(0, np.round(raw_preds - 1.96 * pred_std, 0))
        pred_df['Макс. прогноз'] = np.round(raw_preds + 1.96 * pred_std, 0)
        
        # Рейтинг
        max_pred = pred_df['Прогноз продаж (30 дней, шт.)'].max()
        pred_df['Рейтинг успеха (%)'] = np.round(
            (pred_df['Прогноз продаж (30 дней, шт.)'] / max_pred * 100) if max_pred > 0 else 0, 0
        )
        
        # Категории
        def categorize(rating):
            if rating >= 80: return "🔥 Хит продаж"
            elif rating >= 60: return "⭐ Хорошие продажи"
            elif rating >= 40: return "📈 Средние продажи"
            else: return "🔧 Требует внимания"
        
        pred_df['Категория'] = pred_df['Рейтинг успеха (%)'].apply(categorize)
        
        # Потенциальная выручка
        if 'Price' in product_data:
            pred_df['Потенциальная выручка'] = pred_df['Прогноз продаж (30 дней, шт.)'] * product_data['Price']
        
        # Переименование и сортировка
        pred_df.rename(columns={'Magazin': 'Бутик'}, inplace=True)
        return pred_df.sort_values('Прогноз продаж (30 дней, шт.)', ascending=False)
        
    except Exception as e:
        st.error(f"Ошибка прогнозирования: {str(e)}")
        return pd.DataFrame()

def create_visualizations(predictions_df):
    """Визуализации"""
    if predictions_df.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Топ-10 магазинов
        top_10 = predictions_df.head(10)
        fig1 = px.bar(top_10, x='Прогноз продаж (30 дней, шт.)', y='Бутик', 
                     orientation='h', title="Топ-10 магазинов", 
                     color='Рейтинг успеха (%)', color_continuous_scale='viridis')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Распределение по категориям
        category_counts = predictions_df['Категория'].value_counts()
        fig2 = px.pie(values=category_counts.values, names=category_counts.index,
                     title="Распределение по категориям")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# Инициализация состояния
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Основной интерфейс
dataset_file = st.file_uploader("💖 Загрузите файл с данными", type=["csv", "xlsx", "xls"])

if dataset_file:
    df_raw = load_data(dataset_file)
    
    if df_raw is not None:
        # Информация о файле
        with st.expander("📊 Информация о файле"):
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Строк", len(df_raw))
            with col2: st.metric("Колонок", len(df_raw.columns))
            with col3: st.metric("Размер (MB)", f"{dataset_file.size / (1024*1024):.2f}")
            st.dataframe(df_raw.head(), use_container_width=True)
        
        # Настройка колонок
        st.subheader("🔧 Настройка данных")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            cols = st.columns(5)
            with cols[0]: art_col = st.selectbox("Артикул", df_raw.columns, index=0)
            with cols[1]: magazin_col = st.selectbox("Магазин", df_raw.columns, index=min(1, len(df_raw.columns)-1))
            with cols[2]: date_col = st.selectbox("Дата", df_raw.columns, index=min(2, len(df_raw.columns)-1))
            with cols[3]: qty_col = st.selectbox("Количество", df_raw.columns, index=min(3, len(df_raw.columns)-1))
            with cols[4]: price_col = st.selectbox("Цена", df_raw.columns, index=min(4, len(df_raw.columns)-1))
        
        with col2:
            required_cols = [art_col, magazin_col, date_col, qty_col, price_col]
            available_cols = [col for col in df_raw.columns if col not in required_cols]
            cat_features = st.multiselect("Доп. признаки", available_cols)
        
        # Обработка данных
        if st.button("🚀 Обработать и обучить", type="primary"):
            if len(df_raw) < 10:
                st.error("Слишком мало данных (минимум 10 записей)")
            else:
                with st.spinner("Обрабатываем данные..."):
                    df_agg, stats = process_data(df_raw, art_col, magazin_col, date_col, 
                                               qty_col, price_col, cat_features)
                    
                    if len(df_agg) > 0:
                        # Статистика обработки
                        st.success("✅ Данные обработаны!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Записей для обучения", stats['final_rows'])
                        with col2: st.metric("Удалено строк", f"{stats['removed_rows']} ({stats['removed_pct']:.1f}%)")
                        with col3: st.metric("Плохие даты", f"{stats['bad_dates']} ({stats['bad_dates_pct']:.1f}%)")
                        with col4: st.metric("Уникальных товаров", stats['unique_products'])
                        
                        # Обучение модели
                        with st.spinner("Обучаем модель..."):
                            model, features, metrics = train_model(df_agg, cat_features)
                            
                            # Сохранение в сессию
                            st.session_state.update({
                                'df_agg': df_agg, 'model': model, 'features': features,
                                'metrics': metrics, 'cat_features': cat_features, 'processed': True
                            })
                            
                            st.success("🎯 Модель обучена!")
                            col1, col2 = st.columns(2)
                            with col1: st.metric("MAE", f"{metrics['MAE']:.2f}")
                            with col2: st.metric("R²", f"{metrics['R2']:.3f}")
                    else:
                        st.error("Не осталось данных после обработки!")

# Прогнозирование
if st.session_state.get('processed', False):
    st.divider()
    
    # Форма прогнозирования
    submitted, product_data = create_prediction_form(
        st.session_state.cat_features, st.session_state.df_agg
    )
    
    if submitted:
        predictions_df = make_predictions(
            st.session_state.model, st.session_state.features, 
            product_data, st.session_state.df_agg
        )
        
        if not predictions_df.empty:
            st.success("🎉 Прогноз готов!")
            
            # Результаты прогнозирования - ИСПРАВЛЕНА ЛОГИКА
            st.subheader("📈 Результаты прогнозирования")
            
            # Топ-3 с корректными названиями колонок
            top_3 = predictions_df.head(3)
            medals = ["🥇", "🥈", "🥉"]
            
            cols = st.columns(3)
            for i, (_, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.metric(
                        f"{medals[i]} {row['Бутик']}",
                        f"{int(row['Прогноз продаж (30 дней, шт.)'])} шт.",
                        f"{int(row['Рейтинг успеха (%)'])}%"
                    )
            
            # Полная таблица
            display_cols = ['Бутик', 'Прогноз продаж (30 дней, шт.)', 'Рейтинг успеха (%)', 
                          'Категория', 'Мин. прогноз', 'Макс. прогноз']
            if 'Потенциальная выручка' in predictions_df.columns:
                display_cols.append('Потенциальная выручка')
            
            st.dataframe(predictions_df[display_cols], use_container_width=True, hide_index=True)
            
            # Визуализации
            create_visualizations(predictions_df)
            
            # Скачивание
            csv = predictions_df.to_csv(index=False)
            st.download_button("💾 Скачать прогноз", csv, "forecast.csv", "text/csv")

# Справка
with st.sidebar:
    st.header("ℹ️ Справка")
    st.markdown("""
    **Использование:**
    1. Загрузите файл (CSV/Excel)
    2. Настройте колонки
    3. Обработайте данные
    4. Создайте прогноз
    
    **Требования:**
    - Минимум 10 записей
    - Колонки: артикул, магазин, дата, количество, цена
    """)
    
    if st.session_state.get('processed', False):
        st.success("✅ Готово к прогнозам!")
        st.info(f"R² = {st.session_state.metrics['R2']:.3f}")
