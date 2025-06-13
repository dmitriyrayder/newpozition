import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna
import logging
from typing import Dict, List, Tuple, Optional
import warnings

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

def safe_index_selection(columns: List[str], default_index: int = 0) -> int:
    """Безопасный выбор индекса колонки"""
    if not columns:
        return 0
    return min(default_index, len(columns) - 1)

@st.cache_data
def load_data(file) -> Optional[pd.DataFrame]:
    """Безопасная загрузка данных"""
    try:
        # Проверка размера файла (максимум 50MB)
        if hasattr(file, 'size') and file.size > 50 * 1024 * 1024:
            st.error("Файл слишком большой! Максимальный размер: 50MB")
            return None
            
        if file.name.endswith('.csv'):
            # Пробуем разные кодировки
            for encoding in ['utf-8', 'cp1251', 'latin1']:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    logger.info(f"CSV файл загружен с кодировкой {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("Не удалось определить кодировку CSV файла")
            return None
        else:
            df = pd.read_excel(file, engine='openpyxl')
            logger.info("Excel файл успешно загружен")
            return df
            
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        logger.error(f"Ошибка загрузки файла: {e}")
        return None

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
    
    # Обработка дат с улучшенной диагностикой
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    bad_date_rows = df['date'].isna().sum()
    df.dropna(subset=['date'], inplace=True)
    
    # Очистка ключевых колонок
    crucial_cols = ['Qty', 'Art', 'Magazin', 'Price']
    df.dropna(subset=crucial_cols, inplace=True)
    
    # Фильтрация аномальных значений
    df = df[df['Qty'] > 0]  # Количество должно быть положительным
    df = df[df['Price'] > 0]  # Цена должна быть положительной
    
    # Удаление выбросов по количеству (квантили 1% и 99%)
    qty_q1, qty_q99 = df['Qty'].quantile([0.01, 0.99])
    df = df[(df['Qty'] >= qty_q1) & (df['Qty'] <= qty_q99)]
    
    # Удаление выбросов по цене
    price_q1, price_q99 = df['Price'].quantile([0.01, 0.99])
    df = df[(df['Price'] >= price_q1) & (df['Price'] <= price_q99)]
    
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
    
    # Агрегация данных
    agg_logic = {'Qty': 'sum', 'Price': 'mean'}
    for cat_col in cat_features:
        if cat_col in df_30_days.columns:
            agg_logic[cat_col] = 'first'
    
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)
    
    # Статистика обработки
    stats = {
        "total_rows": initial_rows,
        "final_rows": len(df_agg),
        "bad_date_rows": bad_date_rows,
        "outliers_removed": initial_rows - len(df_30_days),
        "unique_products": df_agg['Art'].nunique(),
        "unique_stores": df_agg['Magazin'].nunique()
    }
    
    logger.info(f"Данные обработаны: {initial_rows} -> {len(df_agg)} строк")
    
    return df_agg, stats

@st.cache_resource
def train_model_with_optuna(
    _df_agg: pd.DataFrame, 
    cat_features: Tuple[str, ...],
    n_trials: int = 50
) -> Tuple[CatBoostRegressor, List[str], Dict]:
    """Улучшенное обучение модели с Optuna"""
    
    cat_features_list = list(cat_features)
    target = 'Qty_30_days'
    features = ['Magazin', 'Price'] + cat_features_list
    
    # Подготовка данных
    df_processed = _df_agg[features + [target]].copy()
    all_cat_features = ['Magazin'] + cat_features_list
    
    # Преобразование категориальных признаков
    for col in all_cat_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
    
    X, y = df_processed[features], df_processed[target]
    
    # Проверка размера данных
    if len(X) < 50:
        st.warning("⚠️ Мало данных для качественного обучения модели (рекомендуется >50 записей)")
    
    # Разделение данных
    test_size = min(0.25, max(0.1, 20 / len(X)))  # Адаптивный размер тестовой выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Оптимизация гиперпараметров
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': False,
            'random_seed': 42
        }
        
        try:
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train, 
                cat_features=all_cat_features,
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
        status_text.text(f"Оптимизация: {i+1}/{n_trials} попыток")
    
    progress_bar.empty()
    status_text.empty()
    
    # Обучение финальной модели
    best_params = study.best_params
    best_params['verbose'] = False
    best_params['random_seed'] = 42
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X, y, cat_features=all_cat_features, verbose=False)
    
    # Метрики качества
    test_preds = final_model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, test_preds),
        'R2': r2_score(y_test, test_preds),
        'best_params': best_params
    }
    
    logger.info(f"Модель обучена. MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.2f}")
    
    return final_model, features, metrics

def create_prediction_form(cat_features: List[str], df_agg: pd.DataFrame) -> Tuple[bool, Dict]:
    """Создание улучшенной формы для прогнозирования"""
    
    st.subheader("✍️ Опишите характеристики новой модели")
    
    with st.form("prediction_form"):
        new_product_data = {}
        
        # Динамическое создание колонок
        n_features = len(cat_features)
        n_cols = min(3, max(1, n_features))
        cols = st.columns(n_cols)
        
        # Поля для категориальных признаков
        for i, feature in enumerate(cat_features):
            with cols[i % n_cols]:
                if feature in df_agg.columns:
                    # Получаем топ значений для подсказки
                    top_values = df_agg[feature].value_counts().head(5).index.tolist()
                    help_text = f"Популярные: {', '.join(map(str, top_values[:3]))}"
                    placeholder = f"Например: {top_values[0]}" if top_values else "Введите значение"
                    
                    new_product_data[feature] = st.text_input(
                        f"{feature} ✨",
                        help=help_text,
                        placeholder=placeholder
                    )
                else:
                    new_product_data[feature] = st.text_input(f"{feature} ✨")
        
        # Поле для цены
        if 'Price' in df_agg.columns:
            price_stats = df_agg['Price'].describe()
            price_mean = float(price_stats['mean'])
            price_min = float(price_stats['min'])
            price_max = float(price_stats['max'])
            
            with cols[n_features % n_cols]:
                new_product_data['Price'] = st.number_input(
                    "Цена 💰",
                    min_value=0.0,
                    value=price_mean,
                    step=max(1.0, price_mean * 0.05),
                    format="%.2f",
                    help=f"Диапазон цен в данных: {price_min:.0f} - {price_max:.0f}"
                )
        
        submitted = st.form_submit_button("🔮 Создать прогноз продаж!")
        
        return submitted, new_product_data

def make_predictions(
    model: CatBoostRegressor, 
    features: List[str], 
    new_product_data: Dict, 
    df_agg: pd.DataFrame
) -> pd.DataFrame:
    """Создание прогнозов для всех магазинов"""
    
    # Валидация входных данных
    for key, value in new_product_data.items():
        if key != 'Price' and (pd.isna(value) or str(value).strip() == ""):
            st.error(f"⚠️ Поле '{key}' не может быть пустым!")
            return pd.DataFrame()
    
    # Создание данных для прогноза
    magaziny = df_agg['Magazin'].unique()
    predictions_data = []
    
    for magazin in magaziny:
        row = new_product_data.copy()
        row['Magazin'] = magazin
        predictions_data.append(row)
    
    predictions_df = pd.DataFrame(predictions_data)[features]
    
    # Получение прогнозов
    try:
        raw_predictions = model.predict(predictions_df)
        predictions_df['Pred_Qty_30_days'] = np.maximum(0, np.round(raw_predictions, 0))
        
        # Расчет рейтинга
        max_pred = predictions_df['Pred_Qty_30_days'].max()
        if max_pred > 0:
            predictions_df['Rating_%'] = np.round(
                (predictions_df['Pred_Qty_30_days'] / max_pred * 100), 0
            )
        else:
            predictions_df['Rating_%'] = 0
        
        # Сортировка по убыванию прогноза
        result_df = predictions_df.sort_values(
            by='Pred_Qty_30_days', ascending=False
        ).rename(columns={
            'Magazin': 'Бутик',
            'Pred_Qty_30_days': 'Прогноз продаж (30 дней, шт.)',
            'Rating_%': 'Рейтинг успеха (%)'
        })
        
        return result_df
        
    except Exception as e:
        st.error(f"Ошибка при создании прогноза: {str(e)}")
        logger.error(f"Ошибка прогнозирования: {e}")
        return pd.DataFrame()

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
                st.metric("Размер (MB)", f"{dataset_file.size / (1024*1024) }
