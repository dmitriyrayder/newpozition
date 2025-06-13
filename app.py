import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# --- СТИЛИЗАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Модный Советник", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
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
.metric-card { padding: 10px; border-radius: 10px; background-color: #fff1f8; border: 1px solid #f8bbd0; }
.prediction-card { 
    background: linear-gradient(135deg, #fff1f8 0%, #fce4ec 100%);
    padding: 20px; border-radius: 15px; border: 2px solid #f8bbd0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("💖 Модный Советник по Продажам 💖")

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Валидация входных данных"""
    if df.empty:
        st.error("Файл пуст! Загрузите файл с данными.")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"В файле отсутствуют колонки: {missing_cols}")
        return False
    
    if len(df) < 10:
        st.warning("Слишком мало данных для качественного анализа (минимум 10 записей)")
        return False
    
    return True

def safe_index_selection(columns, default_index: int = 0) -> int:
    """Безопасный выбор индекса колонки"""
    if len(columns) == 0:
        return 0
    return min(default_index, len(columns) - 1)

@st.cache_data
def load_data(file) -> Optional[pd.DataFrame]:
    """Улучшенная загрузка данных с обработкой дат"""
    try:
        # Проверка размера файла (максимум 50MB)
        if hasattr(file, 'size') and file.size > 50 * 1024 * 1024:
            st.error("Файл слишком большой! Максимальный размер: 50MB")
            return None
            
        if file.name.endswith('.csv'):
            # Пробуем разные кодировки
            for encoding in ['utf-8', 'cp1251', 'latin1', 'utf-8-sig']:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    logger.info(f"CSV файл загружен с кодировкой {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("Не удалось определить кодировку CSV файла")
            return None
        else:
            # Для Excel файлов с принудительным типом даты
            try:
                df = pd.read_excel(file, engine='openpyxl')
                logger.info("Excel файл успешно загружен")
                return df
            except Exception as e:
                # Попытка с другими движками
                try:
                    df = pd.read_excel(file, engine='xlrd')
                    logger.info("Excel файл загружен с xlrd")
                    return df
                except:
                    raise e
            
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        logger.error(f"Ошибка загрузки файла: {e}")
        return None

def parse_dates_robust(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Робустная обработка дат с принудительным преобразованием"""
    df = df.copy()
    
    # Сначала пытаемся стандартное преобразование
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Если много NaT, пробуем разные форматы
    if df[date_col].isna().sum() > len(df) * 0.1:  # Если >10% не распознались
        st.warning("⚠️ Обнаружены проблемы с форматом дат. Применяем дополнительную обработку...")
        
        # Пробуем различные форматы
        date_formats = [
            '%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S',
            '%d-%m-%Y', '%Y/%m/%d'
        ]
        
        original_col = f"{date_col}_original"
        df[original_col] = df[date_col]
        
        for fmt in date_formats:
            mask = df[date_col].isna()
            if mask.sum() == 0:
                break
            try:
                df.loc[mask, date_col] = pd.to_datetime(
                    df.loc[mask, original_col], format=fmt, errors='coerce'
                )
            except:
                continue
        
        df.drop(columns=[original_col], inplace=True)
    
    return df

@st.cache_data
def process_and_aggregate(
    _df: pd.DataFrame, 
    art_col: str, 
    magazin_col: str, 
    date_col: str, 
    qty_col: str, 
    price_col: str, 
    cat_features: Tuple[str, ...]
) -> Tuple[pd.DataFrame, Dict]:
    """Улучшенная обработка и агрегация данных"""
    
    df = _df.copy()
    
    # Переименование колонок
    column_map = {
        art_col: 'Art', 
        magazin_col: 'Magazin', 
        date_col: 'date', 
        qty_col: 'Qty', 
        price_col: 'Price'
    }
    df.rename(columns=column_map, inplace=True)
    
    initial_rows = len(df)
    
    # Робастная обработка дат
    df = parse_dates_robust(df, 'date')
    bad_date_rows = df['date'].isna().sum()
    df.dropna(subset=['date'], inplace=True)
    
    # Очистка ключевых колонок
    crucial_cols = ['Qty', 'Art', 'Magazin', 'Price']
    df.dropna(subset=crucial_cols, inplace=True)
    
    # Фильтрация аномальных значений
    df = df[df['Qty'] > 0]  # Количество должно быть положительным
    df = df[df['Price'] > 0]  # Цена должна быть положительной
    
    # Улучшенное удаление выбросов с использованием IQR
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    df = remove_outliers_iqr(df, 'Qty')
    df = remove_outliers_iqr(df, 'Price')
    
    # Сортировка и группировка
    df = df.sort_values(by=['Art', 'Magazin', 'date'])
    
    # Находим первую дату продажи для каждой пары товар-магазин
    first_sale_dates = df.groupby(['Art', 'Magazin'])['date'].first().reset_index()
    first_sale_dates.rename(columns={'date': 'first_sale_date'}, inplace=True)
    
    # Объединяем с основными данными
    df_merged = pd.merge(df, first_sale_dates, on=['Art', 'Magazin'])
    
    # Берем данные за первые 30 дней
    df_30_days = df_merged[
        df_merged['date'] <= (df_merged['first_sale_date'] + pd.Timedelta(days=30))
    ].copy()
    
    # Добавляем дополнительные признаки
    df_30_days['days_since_launch'] = (df_30_days['date'] - df_30_days['first_sale_date']).dt.days
    df_30_days['revenue'] = df_30_days['Qty'] * df_30_days['Price']
    
    # Агрегация данных с дополнительными метриками
    agg_logic = {
        'Qty': ['sum', 'mean', 'std'],
        'Price': ['mean', 'std'],
        'revenue': 'sum',
        'days_since_launch': 'max'
    }
    
    for cat_col in cat_features:
        if cat_col in df_30_days.columns:
            agg_logic[cat_col] = 'first'
    
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    
    # Сглаживание названий колонок
    df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]
    df_agg.rename(columns={
        'Qty_sum': 'Qty_30_days',
        'Qty_mean': 'Avg_daily_qty',
        'Qty_std': 'Qty_volatility',
        'Price_mean': 'Price',
        'Price_std': 'Price_volatility',
        'revenue_sum': 'Total_revenue_30_days',
        'days_since_launch_max': 'Days_in_sale'
    }, inplace=True)
    
    # Заполняем NaN значения
    df_agg['Qty_volatility'] = df_agg['Qty_volatility'].fillna(0)
    df_agg['Price_volatility'] = df_agg['Price_volatility'].fillna(0)
    
    # Статистика обработки
    stats = {
        "total_rows": initial_rows,
        "final_rows": len(df_agg),
        "bad_date_rows": bad_date_rows,
        "outliers_removed": initial_rows - len(df_30_days),
        "unique_products": df_agg['Art'].nunique(),
        "unique_stores": df_agg['Magazin'].nunique(),
        "date_range": {
            "start": df['date'].min(),
            "end": df['date'].max()
        },
        "avg_price": df_agg['Price'].mean(),
        "total_revenue": df_agg['Total_revenue_30_days'].sum()
    }
    
    logger.info(f"Данные обработаны: {initial_rows} -> {len(df_agg)} строк")
    
    return df_agg, stats

@st.cache_resource
def train_model_with_optuna(
    _df_agg: pd.DataFrame, 
    cat_features: Tuple[str, ...],
    n_trials: int = 50
) -> Tuple[CatBoostRegressor, List[str], Dict]:
    """Улучшенное обучение модели с расширенным набором признаков"""
    
    cat_features_list = list(cat_features)
    target = 'Qty_30_days'
    
    # Расширенный набор признаков
    base_features = ['Magazin', 'Price', 'Avg_daily_qty', 'Price_volatility', 'Days_in_sale']
    features = base_features + cat_features_list
    
    # Фильтруем признаки, которые есть в данных
    available_features = [f for f in features if f in _df_agg.columns]
    
    # Подготовка данных
    df_processed = _df_agg[available_features + [target]].copy()
    all_cat_features = ['Magazin'] + cat_features_list
    
    # Преобразование категориальных признаков
    for col in all_cat_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
    
    X, y = df_processed[available_features], df_processed[target]
    
    # Проверка размера данных
    if len(X) < 50:
        st.warning("⚠️ Мало данных для качественного обучения модели (рекомендуется >50 записей)")
    
    # Адаптивное разделение данных
    test_size = min(0.25, max(0.1, 20 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=None
    )
    
    # Оптимизация гиперпараметров
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'verbose': False,
            'random_seed': 42,
            'loss_function': 'RMSE'
        }
        
        try:
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train, 
                cat_features=[f for f in all_cat_features if f in available_features],
                eval_set=(X_test, y_test),
                early_stopping_rounds=50,
                use_best_model=True,
                verbose=False
            )
            predictions = model.predict(X_test)
            return mean_absolute_error(y_test, predictions)
        except Exception as e:
            logger.error(f"Ошибка в objective функции: {e}")
            return float('inf')
    
    # Обучение с прогресс-баром
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    
    for i in range(n_trials):
        study.optimize(objective, n_trials=1)
        progress = (i + 1) / n_trials
        progress_bar.progress(progress)
        status_text.text(f"Оптимизация: {i+1}/{n_trials} попыток (лучший MAE: {study.best_value:.2f})")
    
    progress_bar.empty()
    status_text.empty()
    
    # Обучение финальной модели
    best_params = study.best_params
    best_params['verbose'] = False
    best_params['random_seed'] = 42
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(
        X, y, 
        cat_features=[f for f in all_cat_features if f in available_features], 
        verbose=False
    )
    
    # Расширенные метрики качества
    test_preds = final_model.predict(X_test)
    train_preds = final_model.predict(X_train)
    
    metrics = {
        'MAE_test': mean_absolute_error(y_test, test_preds),
        'MAE_train': mean_absolute_error(y_train, train_preds),
        'RMSE_test': np.sqrt(mean_squared_error(y_test, test_preds)),
        'R2_test': r2_score(y_test, test_preds),
        'R2_train': r2_score(y_train, train_preds),
        'best_params': best_params,
        'feature_importance': dict(zip(available_features, final_model.feature_importances_)),
        'overfit_ratio': mean_absolute_error(y_train, train_preds) / mean_absolute_error(y_test, test_preds)
    }
    
    logger.info(f"Модель обучена. MAE: {metrics['MAE_test']:.2f}, R²: {metrics['R2_test']:.2f}")
    
    return final_model, available_features, metrics

def create_prediction_form(cat_features: List[str], df_agg: pd.DataFrame) -> Tuple[bool, Dict]:
    """Улучшенная форма для прогнозирования с валидацией"""
    
    st.subheader("✍️ Опишите характеристики новой модели")
    
    with st.form("prediction_form"):
        new_product_data = {}
        
        # Создаем две колонки для лучшего размещения
        col1, col2 = st.columns(2)
        
        # Левая колонка - категориальные признаки
        with col1:
            st.markdown("**Характеристики товара:**")
            for feature in cat_features:
                if feature in df_agg.columns:
                    top_values = df_agg[feature].value_counts().head(5).index.tolist()
                    help_text = f"Популярные: {', '.join(map(str, top_values[:3]))}" if top_values else None
                    placeholder = f"Например: {top_values[0]}" if top_values else "Введите значение"
                    
                    new_product_data[feature] = st.text_input(
                        f"{feature} ✨",
                        help=help_text,
                        placeholder=placeholder,
                        key=f"input_{feature}"
                    )
        
        # Правая колонка - численные параметры
        with col2:
            st.markdown("**Параметры продаж:**")
            
            # Поле для цены
            if 'Price' in df_agg.columns:
                price_stats = df_agg['Price'].describe()
                price_mean = float(price_stats['mean'])
                price_min = max(1.0, float(price_stats['min']))
                price_max = float(price_stats['max'])
                
                new_product_data['Price'] = st.number_input(
                    "Цена 💰",
                    min_value=price_min,
                    max_value=price_max * 2,
                    value=price_mean,
                    step=max(1.0, price_mean * 0.05),
                    format="%.2f",
                    help=f"Диапазон цен в данных: {price_min:.0f} - {price_max:.0f}"
                )
            
            # Дополнительные параметры для более точного прогноза
            if 'Avg_daily_qty' in df_agg.columns:
                avg_daily_mean = float(df_agg['Avg_daily_qty'].mean())
                new_product_data['Avg_daily_qty'] = st.number_input(
                    "Ожидаемые ежедневные продажи",
                    min_value=0.1,
                    value=avg_daily_mean,
                    step=0.1,
                    format="%.1f",
                    help="Предполагаемое среднее количество продаж в день"
                )
            
            new_product_data['Days_in_sale'] = st.slider(
                "Дни в продаже",
                min_value=1,
                max_value=30,
                value=30,
                help="Количество дней, которое товар будет в продаже"
            )
            
            new_product_data['Price_volatility'] = st.number_input(
                "Волатильность цены",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help="Стандартное отклонение цены (0 = стабильная цена)"
            )
        
        # Кнопка прогнозирования
        submitted = st.form_submit_button(
            "🔮 Создать прогноз продаж!",
            type="primary",
            use_container_width=True
        )
        
        return submitted, new_product_data

def make_predictions(
    model: CatBoostRegressor, 
    features: List[str], 
    new_product_data: Dict, 
    df_agg: pd.DataFrame
) -> pd.DataFrame:
    """Улучшенное создание прогнозов с доверительными интервалами"""
    
    # Валидация входных данных
    for key, value in new_product_data.items():
        if key not in ['Price', 'Avg_daily_qty', 'Days_in_sale', 'Price_volatility']:
            if pd.isna(value) or str(value).strip() == "":
                st.error(f"⚠️ Поле '{key}' не может быть пустым!")
                return pd.DataFrame()
    
    # Создание данных для прогноза
    magaziny = df_agg['Magazin'].unique()
    predictions_data = []
    
    for magazin in magaziny:
        row = new_product_data.copy()
        row['Magazin'] = magazin
        predictions_data.append(row)
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Фильтруем только доступные признаки
    available_features = [f for f in features if f in predictions_df.columns]
    predictions_df_filtered = predictions_df[available_features]
    
    # Получение прогнозов
    try:
        raw_predictions = model.predict(predictions_df_filtered)
        predictions_df['Pred_Qty_30_days'] = np.maximum(0, np.round(raw_predictions, 0))
        
        # Расчет доверительного интервала (простой подход)
        predictions_std = raw_predictions.std()
        predictions_df['Pred_Min'] = np.maximum(0, np.round(raw_predictions - 1.96 * predictions_std, 0))
        predictions_df['Pred_Max'] = np.round(raw_predictions + 1.96 * predictions_std, 0)
        
        # Расчет рейтинга и категорий
        max_pred = predictions_df['Pred_Qty_30_days'].max()
        if max_pred > 0:
            predictions_df['Rating_%'] = np.round(
                (predictions_df['Pred_Qty_30_days'] / max_pred * 100), 0
            )
        else:
            predictions_df['Rating_%'] = 0
        
        # Категории магазинов
        def categorize_performance(rating):
            if rating >= 80:
                return "🔥 Хит продаж"
            elif rating >= 60:
                return "⭐ Хорошие продажи"
            elif rating >= 40:
                return "📈 Средние продажи"
            else:
                return "🔧 Требует внимания"
        
        predictions_df['Category'] = predictions_df['Rating_%'].apply(categorize_performance)
        
        # Расчет потенциальной выручки
        if 'Price' in new_product_data:
            predictions_df['Potential_Revenue'] = predictions_df['Pred_Qty_30_days'] * new_product_data['Price']
        
        # Сортировка по убыванию прогноза
        result_df = predictions_df.sort_values(
            by='Pred_Qty_30_days', ascending=False
        )
        
        # Переименование колонок для отображения
        rename_dict = {
            'Magazin': 'Бутик',
            'Pred_Qty_30_days': 'Прогноз продаж (шт.)',
            'Rating_%': 'Рейтинг (%)',
            'Category': 'Категория',
            'Pred_Min': 'Мин. прогноз',
            'Pred_Max': 'Макс. прогноз'
        }
        
        if 'Potential_Revenue' in result_df.columns:
            rename_dict['Potential_Revenue'] = 'Потенциальная выручка'
        
        result_df = result_df.rename(columns=rename_dict)
        
        return result_df
        
    except Exception as e:
        st.error(f"Ошибка при создании прогноза: {str(e)}")
        logger.error(f"Ошибка прогнозирования: {e}")
        return pd.DataFrame()

def create_visualizations(predictions_df: pd.DataFrame, metrics: Dict):
    """Создание визуализаций результатов"""
    
    if predictions_df.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # График топ-10 магазинов
        top_10 = predictions_df.head(10)
        fig1 = px.bar(
            top_10, 
            x='Прогноз продаж (шт.)', 
            y='Бутик',
            orientation='h',
            title="Топ-10 магазинов по прогнозу продаж",
            color='Рейтинг (%)',
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Распределение по категориям
        category_counts = predictions_df['Категория'].value_counts()
        fig2 = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Распределение магазинов по категориям"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # График важности признаков
    if 'feature_importance' in metrics:
        importance_df = pd.DataFrame(
            list(metrics['feature_importance'].items()),
            columns=['Признак', 'Важность']
        ).sort_values('Важность', ascending=True)
        
        fig3 = px.bar(
            importance_df,
            x='Важность',
            y='Признак',
            orientation='h',
            title="Важность признаков в модели"
        )
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ ---

# Инициализация состояния сессии
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Загрузка файла
dataset_file = st.file_uploader(
    "💖 Загрузите файл с данными о продажах",
    type=["csv", "xlsx", "xls"],
    help="Поддерживаются форматы: CSV, Excel (xlsx, xls). Максимальный размер: 50MB"
)

if dataset_file:
    # Загрузка данных
    df_raw = load_data(dataset_file)
    
    if df_raw is not None:
        st.session_state.df_raw = df_raw
        
        # Отображение информации о файле
        with st.expander("📊 Информация о загруженном файле", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Строк", len(df_raw))
            with col2:
                st.metric("Колонок", len(df_raw.columns))
            with col3:
                st.metric("Размер (MB)", f"{dataset_file.size / (1024*1024):.2f}")
            
            st.dataframe(df_raw.head(10), use_container_width=True)
        
        # Настройка колонок
        st.subheader("🔧 Настройка данных")
        
        # Основные колонки
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Основные колонки** (обязательно):")
            cols = st.columns(5)
            
            with cols[0]:
                art_col = st.selectbox(
                    "Артикул товара",
                    options=df_raw.columns,
                    index=safe_index_selection(df_raw.columns, 0)
                )
            
            with cols[1]:
                magazin_col = st.selectbox(
                    "Магазин/Бутик",
                    options=df_raw.columns,
                    index=safe_index_selection(df_raw.columns, 1)
                )
            
            with cols[2]:
                date_col = st.selectbox(
                    "Дата продажи",
                    options=df_raw.columns,
                    index=safe_index_selection(df_raw.columns, 2)
                )
            
            with cols[3]:
                qty_col = st.selectbox(
                    "Количество",
                    options=df_raw.columns,
                    index=safe_index_selection(df_raw.columns, 3)
                )
            
            with cols[4]:
                price_col = st.selectbox(
                    "Цена",
                    options=df_raw.columns,
                    index=safe_index_selection(df_raw.columns, 4)
                )
        
        with col2:
            st.write("**Дополнительные признаки** (необязательно):")
            
            # Выбор дополнительных категориальных признаков
            required_cols = [art_col, magazin_col, date_col, qty_col, price_col]
            available_cols = [col for col in df_raw.columns if col not in required_cols]
            
            cat_features = st.multiselect(
                "Выберите категориальные признаки",
                options=available_cols,
                help="Например: размер, цвет, коллекция, бренд, сезон"
            )
        
        # Кнопка обработки данных
        if st.button(
            "🚀 Обработать данные и обучить модель",
            type="primary",
            use_container_width=True
        ):
            # Валидация
            required_columns = [art_col, magazin_col, date_col, qty_col, price_col]
            
            if validate_dataframe(df_raw, required_columns):
                with st.spinner("🔄 Обрабатываем данные..."):
                    # Обработка и агрегация данных
                    df_agg, stats = process_and_aggregate(
                        df_raw, art_col, magazin_col, date_col, 
                        qty_col, price_col, tuple(cat_features)
                    )
                    
                    if len(df_agg) > 0:
                        st.session_state.df_agg = df_agg
                        st.session_state.stats = stats
                        st.session_state.cat_features = cat_features
                        
                        # Отображение статистики
                        st.success("✅ Данные успешно обработаны!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Записей для обучения", stats['final_rows'])
                        with col2:
                            st.metric("Уникальных товаров", stats['unique_products'])
                        with col3:
                            st.metric("Бутиков", stats['unique_stores'])
                        with col4:
                            st.metric("Очищено строк", stats['outliers_removed'])
                        
                        # Обучение модели
                        with st.spinner("🧠 Обучаем модель..."):
                            model, features, metrics = train_model_with_optuna(
                                df_agg, tuple(cat_features), n_trials=30
                            )
                            
                            st.session_state.model = model
                            st.session_state.features = features
                            st.session_state.metrics = metrics
                            st.session_state.processed = True
                            
                            st.success("🎯 Модель успешно обучена!")
                            
                            # Метрики модели
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Средняя ошибка (MAE)", f"{metrics['MAE']:.2f}")
                            with col2:
                                st.metric("Коэффициент детерминации (R²)", f"{metrics['R2']:.3f}")
                    else:
                        st.error("❌ После обработки не осталось данных для анализа!")

# Прогнозирование
if st.session_state.get('processed', False):
    st.divider()
    st.subheader("🔮 Прогнозирование продаж")
    
    # Форма для прогнозирования
    submitted, new_product_data = create_prediction_form(
        st.session_state.cat_features,
        st.session_state.df_agg
    )
    
    if submitted:
        # Создание прогнозов
        predictions_df = make_predictions(
            st.session_state.model,
            st.session_state.features,
            new_product_data,
            st.session_state.df_agg
        )
        
        if not predictions_df.empty:
            st.success("🎉 Прогноз готов!")
            
            # Отображение результатов
            st.subheader("📈 Результаты прогнозирования")
            
            # Топ-3 магазина
            top_3 = predictions_df.head(3)
            
            col1, col2, col3 = st.columns(3)
            for i, (_, row) in enumerate(top_3.iterrows()):
                with [col1, col2, col3][i]:
                    st.metric(
                        f"🥇 {row['Бутик']}" if i == 0 else f"🥈 {row['Бутик']}" if i == 1 else f"🥉 {row['Бутик']}",
                        f"{int(row['Прогноз продаж (30 дней, шт.)'])} шт.",
                        f"{int(row['Рейтинг успеха (%)'])}%"
                    )
            
            # Полная таблица результатов
            st.dataframe(
                predictions_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Кнопка скачивания
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="💾 Скачать прогноз (CSV)",
                data=csv,
                file_name="fashion_sales_forecast.csv",
                mime="text/csv"
            )

# Справка
with st.sidebar:
    st.header("ℹ️ Справка")
    st.markdown("""
    **Как использовать:**
    1. Загрузите файл с данными о продажах
    2. Настройте соответствие колонок
    3. Выберите дополнительные признаки
    4. Нажмите "Обработать данные"
    5. Заполните характеристики товара
    6. Получите прогноз по всем бутикам
    
    **Требования к данным:**
    - Минимум 10 записей
    - Обязательные колонки: артикул, магазин, дата, количество, цена
    - Форматы файлов: CSV, Excel
    """)
    
    if st.session_state.get('processed', False):
        st.success("✅ Модель готова к использованию!")
        st.info(f"Точность модели: R² = {st.session_state.metrics['R2']:.3f}")
