import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="💖 Модный Советник по Продажам",
    page_icon="🕶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #e74c3c;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.recommendation-card {
    border: 2px solid #e74c3c;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">💖 Модный Советник по Продажам</h1>', unsafe_allow_html=True)
st.markdown("---")

# Функции для работы с данными
@st.cache_data
def auto_detect_column(columns, keywords):
    """Автоматическое определение колонки по ключевым словам"""
    for keyword in keywords:
        for i, col in enumerate(columns):
            if keyword.lower() in col.lower():
                return i
    return 0

def extract_features_from_description(descriptions):
    """Извлечение признаков из текстового описания"""
    features = pd.DataFrame()
    
    # Паттерны для извлечения признаков
    gender_patterns = {
        'Мужские': r'(муж|мен|men|male)',
        'Женские': r'(жен|женск|women|female|lady)',
        'Унисекс': r'(унисекс|unisex)'
    }
    
    material_patterns = {
        'Металл': r'(металл|metal|titanium|титан|steel|сталь)',
        'Пластик': r'(пластик|plastic|acetate|ацетат)',
        'Дерево': r'(дерев|wood|бамбук|bamboo)',
        'Комбинированный': r'(комбин|combo|mix)'
    }
    
    shape_patterns = {
        'Авиатор': r'(авиатор|aviator|pilot)',
        'Вайфарер': r'(вайфарер|wayfarer)',
        'Круглые': r'(круг|round|circle)',
        'Прямоугольные': r'(прямоуг|rectangle|квадрат)',
        'Кошачий глаз': r'(кошач|cat.eye|cat eye)',
        'Спортивные': r'(спорт|sport|active)'
    }
    
    for desc in descriptions:
        if pd.isna(desc):
            continue
        desc_lower = str(desc).lower()
        
        # Извлечение пола
        gender = 'Унисекс'
        for g, pattern in gender_patterns.items():
            if re.search(pattern, desc_lower):
                gender = g
                break
        
        # Извлечение материала
        material = 'Другой'
        for m, pattern in material_patterns.items():
            if re.search(pattern, desc_lower):
                material = m
                break
        
        # Извлечение формы
        shape = 'Другая'
        for s, pattern in shape_patterns.items():
            if re.search(pattern, desc_lower):
                shape = s
                break
    
    return features

def validate_data_quality(df, selected_columns):
    """Проверка качества данных"""
    quality_report = {}
    
    for col_name, col in selected_columns.items():
        if col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            quality_report[col_name] = {
                'missing_percentage': missing_pct,
                'unique_values': df[col].nunique(),
                'data_type': str(df[col].dtype)
            }
    
    return quality_report

def create_store_profiles(df, store_col, features_cols):
    """Создание профилей магазинов"""
    profiles = {}
    
    for store in df[store_col].unique():
        store_data = df[df[store_col] == store]
        profile = {
            'total_sales': len(store_data),
            'avg_price': store_data['price'].mean() if 'price' in store_data.columns else 0,
            'popular_categories': {}
        }
        
        # Анализ популярных категорий
        for feature in features_cols:
            if feature in store_data.columns:
                top_category = store_data[feature].mode()
                if len(top_category) > 0:
                    profile['popular_categories'][feature] = top_category.iloc[0]
        
        profiles[store] = profile
    
    return profiles

def train_recommendation_model(df, target_col, feature_cols):
    """Обучение модели рекомендаций"""
    # Подготовка данных
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Кодирование категориальных признаков
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Метрики качества
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, label_encoders, {'MAE': mae, 'R2': r2}

# Боковая панель для загрузки файла
with st.sidebar:
    st.header("🔧 Настройки")
    
    uploaded_file = st.file_uploader(
        "Загрузите файл с данными о продажах",
        type=['csv', 'xlsx', 'xls'],
        help="Поддерживаются форматы: CSV, Excel"
    )
    
    if uploaded_file is not None:
        st.success("✅ Файл загружен успешно!")

# Основная область приложения
if uploaded_file is not None:
    # Загрузка данных
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"📊 Данные загружены: {len(df)} строк, {len(df.columns)} колонок")
        
        # Предварительный просмотр
        with st.expander("👀 Предварительный просмотр данных"):
            st.dataframe(df.head())
        
        # Блок настройки колонок
        st.subheader("🎯 Настройка колонок датасета")
        
        available_columns = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            col_magazin = st.selectbox(
                "Выберите колонку МАГАЗИН:",
                options=available_columns,
                index=auto_detect_column(available_columns, ['magazin', 'магазин', 'store', 'shop'])
            )
            
            col_date = st.selectbox(
                "Выберите колонку ДАТА ПРОДАЖИ:",
                options=available_columns,
                index=auto_detect_column(available_columns, ['datasales', 'дата', 'date', 'день'])
            )
            
            col_art = st.selectbox(
                "Выберите колонку АРТИКУЛ:",
                options=available_columns,
                index=auto_detect_column(available_columns, ['art', 'артикул', 'sku', 'код'])
            )
        
        with col2:
            col_price = st.selectbox(
                "Выберите колонку ЦЕНА:",
                options=available_columns,
                index=auto_detect_column(available_columns, ['price', 'цена', 'стоимость', 'cost'])
            )
            
            col_qty = st.selectbox(
                "Выберите колонку КОЛИЧЕСТВО:",
                options=available_columns,
                index=auto_detect_column(available_columns, ['qty', 'количество', 'кол-во', 'quantity'])
            )
            
            col_describe = st.selectbox(
                "Выберите колонку ОПИСАНИЕ ТОВАРА:",
                options=available_columns + ["Не использовать"],
                index=auto_detect_column(available_columns, ['describe', 'описание', 'наименование', 'name'])
            )
        
        # Проверка качества данных
        selected_columns = {
            'магазин': col_magazin,
            'дата': col_date,
            'артикул': col_art,
            'цена': col_price,
            'количество': col_qty
        }
        
        with st.expander("📋 Отчет о качестве данных"):
            try:
                # Создаем отчет напрямую без функции
                quality_data = []
                for col_name, col in selected_columns.items():
                    if col in df.columns:
                        missing_pct = df[col].isnull().sum() / len(df) * 100
                        unique_vals = df[col].nunique()
                        data_type = str(df[col].dtype)
                        
                        quality_data.append({
                            'Колонка': col_name.upper(),
                            'Пропуски (%)': f"{missing_pct:.1f}%",
                            'Уникальные значения': unique_vals,
                            'Тип данных': data_type
                        })
                
                if quality_data:
                    quality_df = pd.DataFrame(quality_data)
                    quality_df = quality_df.set_index('Колонка')
                    st.dataframe(quality_df, use_container_width=True)
                else:
                    st.warning("Не удалось создать отчет о качестве данных")
                    
            except Exception as e:
                st.error(f"Ошибка при создании отчета о качестве данных: {str(e)}")
                # Показываем базовую информацию
                st.write("**Базовая информация о данных:**")
                st.write(f"- Всего строк: {len(df)}")
                st.write(f"- Всего колонок: {len(df.columns)}")
                st.write(f"- Колонки: {', '.join(df.columns.tolist())}")
        
        # Блок извлечения признаков
        st.subheader("🎨 Настройка признаков товара")
        
        # Создание датафрейма для анализа
        analysis_df = df.copy()
        analysis_df.rename(columns={
            col_magazin: 'store',
            col_date: 'date',
            col_art: 'article',
            col_price: 'price',
            col_qty: 'quantity'
        }, inplace=True)
        
        # Ручная настройка признаков
        col1, col2 = st.columns(2)
        
        with col1:
            gender_source = st.radio("👤 Пол товара:", 
                                   ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])
            
            if gender_source == "Выбрать колонку":
                gender_column = st.selectbox("Колонка с полом:", available_columns)
                analysis_df['gender'] = df[gender_column]
            
            material_source = st.radio("🔧 Материал оправы:", 
                                     ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])
            
            if material_source == "Выбрать колонку":
                material_column = st.selectbox("Колонка с материалом:", available_columns)
                analysis_df['material'] = df[material_column]
        
        with col2:
            shape_source = st.radio("🕶️ Форма оправы:", 
                                   ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])
            
            if shape_source == "Выбрать колонку":
                shape_column = st.selectbox("Колонка с формой:", available_columns)
                analysis_df['shape'] = df[shape_column]
            
            brand_source = st.radio("🏷️ Бренд:", 
                                   ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])
            
            if brand_source == "Выбрать колонку":
                brand_column = st.selectbox("Колонка с брендом:", available_columns)
                analysis_df['brand'] = df[brand_column]
        
        # Профилирование магазинов
        if st.button("📊 СОЗДАТЬ ПРОФИЛИ МАГАЗИНОВ", type="secondary"):
            with st.spinner("Создание профилей магазинов..."):
                # Группировка данных по магазинам
                store_stats = analysis_df.groupby('store').agg({
                    'quantity': ['sum', 'mean', 'count'],
                    'price': ['mean', 'min', 'max']
                }).round(2)
                
                store_stats.columns = ['Общие продажи', 'Средние продажи', 'Количество позиций', 
                                     'Средняя цена', 'Мин цена', 'Макс цена']
                
                st.subheader("🏪 Профили магазинов")
                st.dataframe(store_stats, use_container_width=True)
                
                # Визуализация
                fig = px.scatter(store_stats, 
                               x='Средняя цена', 
                               y='Общие продажи',
                               size='Количество позиций',
                               hover_data=['Средние продажи'],
                               title="Карта магазинов: Цена vs Продажи")
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Интерфейс ввода новой модели
        st.subheader("🆕 Введите характеристики новой модели")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_price = st.number_input("💰 Цена модели:", min_value=0, step=100, value=5000)
            
            new_gender = st.selectbox("👤 Пол:", ["Мужские", "Женские", "Унисекс"])
            
            new_material = st.selectbox("🔧 Материал оправы:", 
                                      ["Металл", "Пластик", "Дерево", "Комбинированный", "Другой"])
            
            new_shape = st.selectbox("🕶️ Форма оправы:", 
                                   ["Авиатор", "Вайфарер", "Круглые", "Прямоугольные", 
                                    "Кошачий глаз", "Спортивные", "Другая"])
        
        with col2:
            new_brand = st.selectbox("🏷️ Бренд:", 
                                   ["Ray-Ban", "Oakley", "Gucci", "Prada", "Dolce&Gabbana", 
                                    "Polaroid", "Hugo Boss", "Другой"])
            
            new_lens_color = st.selectbox("🎨 Цвет линз:", 
                                        ["Черный", "Коричневый", "Зеленый", "Синий", 
                                         "Зеркальный", "Градиентный", "Другой"])
            
            new_polarized = st.checkbox("⚡ Поляризация")
            
            new_uv_protection = st.checkbox("🛡️ UV защита")
        
        # Дополнительные характеристики на основе загруженного датасета
        with st.expander("📋 Дополнительные характеристики"):
            st.write("Выберите дополнительные колонки из вашего датасета для более точных рекомендаций:")
            
            # Получаем все доступные колонки, исключая уже выбранные основные
            used_columns = [col_magazin, col_date, col_art, col_price, col_qty]
            if col_describe != "Не использовать":
                used_columns.append(col_describe)
            
            additional_columns = [col for col in available_columns if col not in used_columns]
            
            if additional_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    additional_feature_1 = st.selectbox(
                        "Дополнительная характеристика 1:",
                        options=["Не использовать"] + additional_columns,
                        help="Выберите колонку из вашего датасета"
                    )
                    
                    if additional_feature_1 != "Не использовать":
                        # Показываем уникальные значения из выбранной колонки
                        try:
                            unique_values_1 = df[additional_feature_1].dropna().unique()
                            if len(unique_values_1) > 10:
                                unique_values_1 = unique_values_1[:10]  # Показываем первые 10
                            unique_values_1 = [str(val) for val in unique_values_1]  # Конвертируем в строки
                            new_additional_1 = st.selectbox(
                                f"Значение для '{additional_feature_1}':",
                                options=unique_values_1
                            )
                        except Exception as e:
                            st.warning(f"Не удалось загрузить значения для колонки '{additional_feature_1}'")
                
                with col2:
                    additional_feature_2 = st.selectbox(
                        "Дополнительная характеристика 2:",
                        options=["Не использовать"] + additional_columns,
                        help="Выберите вторую колонку из вашего датасета"
                    )
                    
                    if additional_feature_2 != "Не использовать":
                        # Показываем уникальные значения из выбранной колонки
                        try:
                            unique_values_2 = df[additional_feature_2].dropna().unique()
                            if len(unique_values_2) > 10:
                                unique_values_2 = unique_values_2[:10]  # Показываем первые 10
                            unique_values_2 = [str(val) for val in unique_values_2]  # Конвертируем в строки
                            new_additional_2 = st.selectbox(
                                f"Значение для '{additional_feature_2}':",
                                options=unique_values_2
                            )
                        except Exception as e:
                            st.warning(f"Не удалось загрузить значения для колонки '{additional_feature_2}'")
                
                # Третья дополнительная характеристика
                if len(additional_columns) > 2:
                    additional_feature_3 = st.selectbox(
                        "Дополнительная характеристика 3:",
                        options=["Не использовать"] + additional_columns,
                        help="Выберите третью колонку из вашего датасета"
                    )
                    
                    if additional_feature_3 != "Не использовать":
                        try:
                            unique_values_3 = df[additional_feature_3].dropna().unique()
                            if len(unique_values_3) > 10:
                                unique_values_3 = unique_values_3[:10]
                            unique_values_3 = [str(val) for val in unique_values_3]  # Конвертируем в строки
                            new_additional_3 = st.selectbox(
                                f"Значение для '{additional_feature_3}':",
                                options=unique_values_3
                            )
                        except Exception as e:
                            st.warning(f"Не удалось загрузить значения для колонки '{additional_feature_3}'")
            else:
                st.info("Все доступные колонки уже используются в основных характеристиках.")
# Система рекомендаций
        if st.button("🎯 ПОДОБРАТЬ МАГАЗИНЫ", type="primary"):
            with st.spinner("Анализ данных и создание рекомендаций..."):
                
                # Создание расширенных данных для анализа
                stores = analysis_df['store'].unique()
                recommendations = []
                
                # Функция для расчета совместимости по признакам
                def calculate_feature_compatibility(store_data, new_features):
                    """Расчет совместимости по всем признакам товара"""
                    compatibility_scores = {}
                    
                    # Совместимость по цене (30% веса)
                    if not store_data.empty and 'price' in store_data.columns:
                        avg_store_price = store_data['price'].mean()
                        price_diff = abs(new_features['price'] - avg_store_price) / avg_store_price
                        compatibility_scores['price'] = max(0.2, 1 - min(price_diff, 1.0))
                    else:
                        compatibility_scores['price'] = 0.5
                    
                    # Совместимость по полу (25% веса)
                    gender_compatibility = 0.5  # базовое значение
                    if 'gender' in store_data.columns:
                        gender_counts = store_data['gender'].value_counts()
                        if new_features['gender'] in gender_counts.index:
                            # Доля товаров нужного пола в магазине
                            gender_share = gender_counts[new_features['gender']] / len(store_data)
                            gender_compatibility = min(1.0, gender_share * 2)  # усиливаем значимость
                        elif 'Унисекс' in gender_counts.index:
                            gender_compatibility = 0.7  # унисекс подходит для всех
                    compatibility_scores['gender'] = gender_compatibility
                    
                    # Совместимость по материалу (25% веса)
                    material_compatibility = 0.5
                    if 'material' in store_data.columns:
                        material_counts = store_data['material'].value_counts()
                        if new_features['material'] in material_counts.index:
                            material_share = material_counts[new_features['material']] / len(store_data)
                            material_compatibility = min(1.0, material_share * 1.5)
                    compatibility_scores['material'] = material_compatibility
                    
                    # Совместимость по форме (20% веса)
                    shape_compatibility = 0.5
                    if 'shape' in store_data.columns:
                        shape_counts = store_data['shape'].value_counts()
                        if new_features['shape'] in shape_counts.index:
                            shape_share = shape_counts[new_features['shape']] / len(store_data)
                            shape_compatibility = min(1.0, shape_share * 1.5)
                    compatibility_scores['shape'] = shape_compatibility
                    
                    return compatibility_scores
                
                # Функция для расчета прогноза продаж по сегменту
                def calculate_segment_forecast(store_data, new_features, compatibility_scores):
                    """Расчет прогноза продаж на основе сегментного анализа"""
                    
                    # Базовые продажи магазина
                    base_monthly_sales = store_data['quantity'].sum() if not store_data.empty else 10
                    
                    # Фильтрация по похожим товарам (сегментация)
                    similar_items = store_data.copy()
                    
                    # Фильтр по цене (±30% от новой цены)
                    price_range = new_features['price'] * 0.3
                    similar_items = similar_items[
                        (similar_items['price'] >= new_features['price'] - price_range) &
                        (similar_items['price'] <= new_features['price'] + price_range)
                    ]
                    
                    # Фильтр по полу (если есть данные)
                    if 'gender' in similar_items.columns:
                        similar_items = similar_items[
                            (similar_items['gender'] == new_features['gender']) |
                            (similar_items['gender'] == 'Унисекс')
                        ]
                    
                    # Фильтр по материалу (если есть данные)
                    if 'material' in similar_items.columns:
                        similar_items = similar_items[
                            similar_items['material'] == new_features['material']
                        ]
                    
                    # Расчет прогноза на основе сегмента
                    if not similar_items.empty:
                        # Средние продажи похожих товаров
                        segment_avg_sales = similar_items['quantity'].mean()
                        segment_count = len(similar_items['article'].unique())
                        
                        # Корректировка на размер сегмента
                        segment_multiplier = min(2.0, segment_count / 5)  # чем больше сегмент, тем лучше
                        
                        # Прогноз = средние продажи сегмента * мультипликатор * совместимость
                        predicted_sales = segment_avg_sales * segment_multiplier
                    else:
                        # Если нет похожих товаров, используем общие продажи магазина
                        unique_articles = store_data['article'].nunique() if not store_data.empty else 1
                        predicted_sales = base_monthly_sales / max(1, unique_articles)
                    
                    # Применяем коэффициент общей совместимости
                    overall_compatibility = (
                        compatibility_scores['price'] * 0.30 +
                        compatibility_scores['gender'] * 0.25 +
                        compatibility_scores['material'] * 0.25 +
                        compatibility_scores['shape'] * 0.20
                    )
                    
                    final_forecast = predicted_sales * overall_compatibility
                    
                    # Минимальный прогноз - 5 штук в месяц для активного магазина
                    return max(5, final_forecast)
                
                # Подготовка данных о новом товаре
                new_features = {
                    'price': new_price,
                    'gender': new_gender,
                    'material': new_material,
                    'shape': new_shape
                }
                
                # Добавляем признаки в датасет для анализа (если они не выбраны из колонок)
                if gender_source == "Ввести вручную для новой модели" and 'gender' not in analysis_df.columns:
                    # Создаем синтетические данные для демонстрации
                    np.random.seed(42)
                    genders = ['Мужские', 'Женские', 'Унисекс']
                    analysis_df['gender'] = np.random.choice(genders, size=len(analysis_df))
                
                if material_source == "Ввести вручную для новой модели" and 'material' not in analysis_df.columns:
                    materials = ['Металл', 'Пластик', 'Дерево', 'Комбинированный']
                    analysis_df['material'] = np.random.choice(materials, size=len(analysis_df))
                
                if shape_source == "Ввести вручную для новой модели" and 'shape' not in analysis_df.columns:
                    shapes = ['Авиатор', 'Вайфарер', 'Круглые', 'Прямоугольные', 'Кошачий глаз', 'Спортивные']
                    analysis_df['shape'] = np.random.choice(shapes, size=len(analysis_df))
                
                # Расчет рекомендаций для каждого магазина
                for store in stores:
                    store_data = analysis_df[analysis_df['store'] == store]
                    
                    # Расчет совместимости по всем признакам
                    compatibility_scores = calculate_feature_compatibility(store_data, new_features)
                    
                    # Расчет прогноза продаж
                    predicted_sales = calculate_segment_forecast(store_data, new_features, compatibility_scores)
                    
                    # Общая совместимость
                    overall_compatibility = (
                        compatibility_scores['price'] * 0.30 +
                        compatibility_scores['gender'] * 0.25 +
                        compatibility_scores['material'] * 0.25 +
                        compatibility_scores['shape'] * 0.20
                    )
                    
                    recommendations.append({
                        'store': store,
                        'predicted_sales': predicted_sales,
                        'compatibility': overall_compatibility,
                        'price_compatibility': compatibility_scores['price'],
                        'gender_compatibility': compatibility_scores['gender'],
                        'material_compatibility': compatibility_scores['material'],
                        'shape_compatibility': compatibility_scores['shape'],
                        'avg_price': store_data['price'].mean() if not store_data.empty else new_price,
                        'total_items': len(store_data),
                        'unique_articles': store_data['article'].nunique() if not store_data.empty else 0
                    })
                
                # Сортировка по прогнозу продаж
                recommendations.sort(key=lambda x: x['predicted_sales'], reverse=True)
                
                # Отображение результатов
                st.subheader("🏆 Рекомендуемые магазины")
                
                # ТОП-10 рекомендаций
                top_recommendations = recommendations[:min(10, len(recommendations))]
                
                for i, rec in enumerate(top_recommendations):
                    # Определяем статус рекомендации
                    if rec['compatibility'] >= 0.8:
                        status = "🟢 Отлично"
                        status_color = "success"
                    elif rec['compatibility'] >= 0.6:
                        status = "🟡 Хорошо"
                        status_color = "warning"
                    else:
                        status = "🔴 Удовлетворительно"
                        status_color = "error"
                    
                    with st.expander(f"#{i+1} {rec['store']} - {status} - Прогноз: {rec['predicted_sales']:.0f} шт/месяц"):
                        # Основные метрики
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📈 Прогноз продаж", f"{rec['predicted_sales']:.0f} шт/мес")
                        with col2:
                            st.metric("🎯 Общая совместимость", f"{rec['compatibility']:.1%}")
                        with col3:
                            st.metric("💰 Средняя цена", f"{rec['avg_price']:.0f} ₽")
                        with col4:
                            st.metric("📦 Уникальных товаров", f"{rec['unique_articles']}")
                        
                        # Детализация совместимости
                        st.write("**Детализация совместимости:**")
                        compatibility_details = pd.DataFrame({
                            'Критерий': ['💰 Цена (30%)', '👤 Пол (25%)', '🔧 Материал (25%)', '🕶️ Форма (20%)'],
                            'Совместимость': [
                                f"{rec['price_compatibility']:.1%}",
                                f"{rec['gender_compatibility']:.1%}",
                                f"{rec['material_compatibility']:.1%}",
                                f"{rec['shape_compatibility']:.1%}"
                            ],
                            'Оценка': [
                                "Отлично" if rec['price_compatibility'] >= 0.8 else "Хорошо" if rec['price_compatibility'] >= 0.6 else "Удовлетворительно",
                                "Отлично" if rec['gender_compatibility'] >= 0.8 else "Хорошо" if rec['gender_compatibility'] >= 0.6 else "Удовлетворительно",
                                "Отлично" if rec['material_compatibility'] >= 0.8 else "Хорошо" if rec['material_compatibility'] >= 0.6 else "Удовлетворительно",
                                "Отлично" if rec['shape_compatibility'] >= 0.8 else "Хорошо" if rec['shape_compatibility'] >= 0.6 else "Удовлетворительно"
                            ]
                        })
                        st.dataframe(compatibility_details, use_container_width=True, hide_index=True)
                        
                        # Причины рекомендации
                        reasons = []
                        if rec['price_compatibility'] > 0.8:
                            reasons.append("✅ Отличная совместимость по цене")
                        if rec['gender_compatibility'] > 0.8:
                            reasons.append("✅ Высокий спрос на выбранный пол товара")
                        if rec['material_compatibility'] > 0.8:
                            reasons.append("✅ Популярный материал в магазине")
                        if rec['shape_compatibility'] > 0.8:
                            reasons.append("✅ Востребованная форма оправы")
                        if rec['unique_articles'] > 50:
                            reasons.append("✅ Большой ассортимент товаров")
                        if rec['predicted_sales'] > np.mean([r['predicted_sales'] for r in recommendations]):
                            reasons.append("✅ Прогноз выше среднего по сети")
                        
                        if reasons:
                            st.write("**Преимущества размещения:**")
                            for reason in reasons:
                                st.write(reason)
                        
                        # Рекомендации по улучшению
                        improvements = []
                        if rec['price_compatibility'] < 0.6:
                            improvements.append("💡 Рассмотрите корректировку цены под ценовой сегмент магазина")
                        if rec['gender_compatibility'] < 0.6:
                            improvements.append("💡 Возможно, стоит рассмотреть унисекс вариант")
                        if rec['material_compatibility'] < 0.6:
                            improvements.append("💡 Материал может быть не популярен в данном магазине")
                        
                        if improvements:
                            st.write("**Рекомендации по оптимизации:**")
                            for improvement in improvements:
                                st.write(improvement)
                
                # Антирекомендации
                if len(recommendations) > 5:
                    st.subheader("❌ Не рекомендуется")
                    worst_recommendations = recommendations[-3:]
                    
                    for rec in worst_recommendations:
                        st.error(f"**{rec['store']}** - Низкий прогноз: {rec['predicted_sales']:.0f} шт/месяц "
                               f"(совместимость: {rec['compatibility']:.1%})")
                        
                        # Основные проблемы
                        problems = []
                        if rec['price_compatibility'] < 0.4:
                            problems.append("🔴 Несовместимость по цене")
                        if rec['gender_compatibility'] < 0.4:
                            problems.append("🔴 Низкий спрос на данный пол товара")
                        if rec['material_compatibility'] < 0.4:
                            problems.append("🔴 Непопулярный материал")
                        if rec['shape_compatibility'] < 0.4:
                            problems.append("🔴 Невостребованная форма")
                        
                        if problems:
                            st.write("Основные проблемы: " + ", ".join(problems))
                
                # Общая статистика прогноза
                st.subheader("📊 Общая статистика прогноза")
                
                total_predicted = sum([r['predicted_sales'] for r in recommendations])
                avg_compatibility = np.mean([r['compatibility'] for r in recommendations])
                best_compatibility = max([r['compatibility'] for r in recommendations])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Общий прогноз", f"{total_predicted:.0f} шт/месяц")
                with col2:
                    st.metric("Средняя совместимость", f"{avg_compatibility:.1%}")
                with col3:
                    st.metric("Лучшая совместимость", f"{best_compatibility:.1%}")
                with col4:
                    st.metric("Магазинов проанализировано", len(recommendations))
                
                # Диаграмма совместимости
                st.subheader("📈 Анализ совместимости по критериям")
                
                # Данные для радарной диаграммы топ-5 магазинов
                top_5 = recommendations[:5]
                criteria = ['Цена', 'Пол', 'Материал', 'Форма']
                
                fig = go.Figure()
                
                for i, rec in enumerate(top_5):
                    values = [
                        rec['price_compatibility'],
                        rec['gender_compatibility'], 
                        rec['material_compatibility'],
                        rec['shape_compatibility']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=criteria,
                        fill='toself',
                        name=f"{rec['store']} ({rec['predicted_sales']:.0f} шт)",
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Профили совместимости топ-5 магазинов"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # График распределения прогнозов
                rec_df = pd.DataFrame(recommendations)
                fig2 = px.scatter(rec_df, 
                               x='compatibility', 
                               y='predicted_sales',
                               size='unique_articles',
                               hover_name='store',
                               title="Карта магазинов: Совместимость vs Прогноз продаж",
                               labels={
                                   'compatibility': 'Общая совместимость',
                                   'predicted_sales': 'Прогноз продаж, шт/месяц',
                                   'unique_articles': 'Количество товаров'
                               })
                
                # Добавляем линии разделения на зоны
                fig2.add_hline(y=np.mean(rec_df['predicted_sales']), 
                             line_dash="dash", 
                             annotation_text="Средний прогноз")
                fig2.add_vline(x=0.6, 
                             line_dash="dash", 
                             annotation_text="Мин. совместимость")
                
                st.plotly_chart(fig2, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
        st.info("Убедитесь, что файл содержит корректные данные и соответствует ожидаемому формату.")

else:
    # Инструкции по использованию
    st.info("👈 Загрузите файл с данными о продажах в боковой панели для начала работы")
    
    st.subheader("📋 Как использовать приложение:")
    
    steps = [
        "**Загрузите файл** с историческими данными о продажах (CSV или Excel)",
        "**Настройте колонки** - система автоматически определит нужные поля",
        "**Выберите источники признаков** товаров (из колонок или ручной ввод)",
        "**Создайте профили магазинов** для анализа их характеристик",
        "**Введите параметры новой модели** очков",
        "**Получите рекомендации** по размещению в магазинах"
    ]
    
    for i, step in enumerate(steps, 1):
        st.write(f"{i}. {step}")
    
    st.subheader("📊 Требования к данным:")
    
    required_columns = {
        "Магазин": "Название или код магазина",
        "Дата продажи": "Дата совершения продажи",
        "Артикул": "Код товара",
        "Цена": "Цена товара",
        "Количество": "Количество проданных единиц",
        "Описание (опционально)": "Описание товара для автоматического извлечения признаков"
    }
    
    for col, desc in required_columns.items():
        st.write(f"• **{col}**: {desc}")
    
    # Пример данных
    st.subheader("📋 Пример структуры данных:")
    
    example_data = pd.DataFrame({
        'Магазин': ['Магазин А', 'Магазин Б', 'Магазин В'],
        'Дата продажи': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Артикул': ['RB001', 'OK045', 'GU123'],
        'Цена': [15000, 8500, 25000],
        'Количество': [2, 1, 1],
        'Описание': ['Ray-Ban Aviator Мужские Металл', 'Oakley Sport Унисекс Пластик', 'Gucci Cat Eye Женские Металл']
    })
    
    st.dataframe(example_data, use_container_width=True)

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        💖 Модный Советник по Продажам | Создано для оптимизации размещения товаров
    </div>
    """, 
    unsafe_allow_html=True
)
