import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(page_title="💖 Модный Советник", layout="wide", initial_sidebar_state="expanded")

# Кастомный CSS для красных кнопок
st.markdown("""
<style>
.stButton > button {
    background-color: #DC143C;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    font-size: 16px;
    padding: 0.5rem 1rem;
    transition: all 0.3s;
}
.stButton > button:hover {
    background-color: #B22222;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def auto_detect_column(columns, keywords):
    """Автоопределение колонок по ключевым словам"""
    for keyword in keywords:
        for i, col in enumerate(columns):
            if keyword.lower() in col.lower():
                return i
    return 0

@st.cache_data
def safe_read_file(uploaded_file):
    """Безопасное чтение файла с обработкой дат"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Попытка автоопределения даты
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'дата', 'datasales'])]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df, len(df), 0
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return None, 0, 1

@st.cache_data
def extract_features_from_text(text_series):
    """Извлечение признаков из текстового описания"""
    features = pd.DataFrame(index=text_series.index)
    
    # Пол
    gender_patterns = {
        'Мужские': r'мужск|male|men',
        'Женские': r'женск|female|women|lady',
        'Унисекс': r'унисекс|unisex'
    }
    features['gender'] = 'Унисекс'
    for gender, pattern in gender_patterns.items():
        mask = text_series.str.contains(pattern, case=False, na=False)
        features.loc[mask, 'gender'] = gender
    
    # Материал
    material_patterns = {
        'Металл': r'металл|metal|steel|титан',
        'Пластик': r'пластик|plastic|acetate',
        'Дерево': r'дерев|wood|bamboo'
    }
    features['material'] = 'Пластик'
    for material, pattern in material_patterns.items():
        mask = text_series.str.contains(pattern, case=False, na=False)
        features.loc[mask, 'material'] = material
    
    # Форма
    shape_patterns = {
        'Авиатор': r'авиатор|aviator|pilot',
        'Вайфарер': r'вайфарер|wayfarer',
        'Круглые': r'кругл|round|circle',
        'Прямоугольные': r'прямоуг|rectangle|square',
        'Кошачий глаз': r'кошач|cat.eye'
    }
    features['shape'] = 'Прямоугольные'
    for shape, pattern in shape_patterns.items():
        mask = text_series.str.contains(pattern, case=False, na=False)
        features.loc[mask, 'shape'] = shape
    
    # Бинарные признаки
    features['is_polarized'] = text_series.str.contains(r'поляр|polar', case=False, na=False).astype(int)
    features['has_uv'] = text_series.str.contains(r'uv|защита|protection', case=False, na=False).astype(int)
    
    return features

@st.cache_data
def create_store_profiles(df, store_col, price_col, date_col, qty_col):
    """Создание профилей магазинов"""
    profiles = {}
    
    for store in df[store_col].unique():
        store_data = df[df[store_col] == store]
        
        profiles[store] = {
            'avg_price': store_data[price_col].mean(),
            'price_std': store_data[price_col].std(),
            'total_sales': store_data[qty_col].sum(),
            'unique_products': store_data.iloc[:, 0].nunique(),  # Предполагаем первая колонка - артикул
            'avg_monthly_sales': store_data.groupby(store_data[date_col].dt.to_period('M'))[qty_col].sum().mean()
        }
    
    return profiles

def prepare_training_data(df, columns_mapping, store_profiles):
    """Подготовка данных для обучения"""
    # Создание признаков товара
    if columns_mapping.get('describe'):
        product_features = extract_features_from_text(df[columns_mapping['describe']])
    else:
        # Если нет описания, создаем базовые признаки
        product_features = pd.DataFrame({
            'gender': 'Унисекс',
            'material': 'Пластик',
            'shape': 'Прямоугольные',
            'is_polarized': 0,
            'has_uv': 0
        }, index=df.index)
    
    # Объединение с основными данными
    train_data = df.copy()
    for col, values in product_features.items():
        train_data[col] = values
    
    # Добавление профилей магазинов
    train_data['store_avg_price'] = train_data[columns_mapping['magazin']].map(
        lambda x: store_profiles.get(x, {}).get('avg_price', 0)
    )
    train_data['store_volume'] = train_data[columns_mapping['magazin']].map(
        lambda x: store_profiles.get(x, {}).get('total_sales', 0)
    )
    
    # Признаки совместимости
    train_data['price_match'] = abs(train_data[columns_mapping['price']] - train_data['store_avg_price']) / train_data['store_avg_price']
    
    # Ценовые сегменты
    train_data['price_segment'] = pd.cut(train_data[columns_mapping['price']], 
                                       bins=[0, 2000, 5000, float('inf')], 
                                       labels=['Эконом', 'Средний', 'Премиум'])
    
    return train_data

@st.cache_resource
def train_model(train_data, target_col, feature_cols, cat_features):
    """Обучение модели CatBoost"""
    X = train_data[feature_cols].copy()
    y = train_data[target_col]
    
    # Обработка пропусков
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')
        else:
            X[col] = X[col].fillna(X[col].median())
    
    model = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        cat_features=cat_features,
        verbose=False,
        random_state=42
    )
    
    model.fit(X, y)
    return model

def predict_for_stores(model, new_product_features, stores, store_profiles, feature_cols):
    """Прогноз для всех магазинов"""
    predictions = {}
    
    for store in stores:
        # Создание строки с признаками для данного магазина
        store_profile = store_profiles.get(store, {})
        
        prediction_row = new_product_features.copy()
        prediction_row.update({
            'store_avg_price': store_profile.get('avg_price', 3000),
            'store_volume': store_profile.get('total_sales', 100),
            'price_match': abs(new_product_features['price'] - store_profile.get('avg_price', 3000)) / store_profile.get('avg_price', 3000),
            'magazin': store
        })
        
        # Создание DataFrame для предсказания
        pred_df = pd.DataFrame([prediction_row])
        
        # Обработка категориальных признаков
        for col in pred_df.columns:
            if col not in feature_cols:
                continue
            if pred_df[col].dtype == 'object':
                pred_df[col] = pred_df[col].fillna('Unknown')
            else:
                pred_df[col] = pred_df[col].fillna(0)
        
        try:
            pred = model.predict(pred_df[feature_cols])[0]
            predictions[store] = max(0, pred)  # Не может быть отрицательных продаж
        except:
            predictions[store] = 0
    
    return predictions

# Основное приложение
def main():
    st.title("💖 Модный Советник по Продажам")
    st.markdown("*Интеллектуальная система подбора магазинов для новых моделей очков*")
    
    # Боковая панель
    with st.sidebar:
        st.header("📁 Загрузка данных")
        uploaded_file = st.file_uploader("Выберите файл с данными", type=['xlsx', 'csv'])
        
        if uploaded_file:
            df, success_rows, error_rows = safe_read_file(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Загружено: {success_rows} строк")
                if error_rows > 0:
                    st.warning(f"⚠️ Ошибки: {error_rows} строк")
                
                # Предварительный просмотр
                with st.expander("👀 Предварительный просмотр"):
                    st.dataframe(df.head(3))
                
                # Настройка колонок
                st.header("🎯 Настройка колонок")
                columns = df.columns.tolist()
                
                columns_mapping = {
                    'magazin': st.selectbox("Магазин:", columns, 
                                          index=auto_detect_column(columns, ['magazin', 'магазин', 'store'])),
                    'date': st.selectbox("Дата:", columns,
                                       index=auto_detect_column(columns, ['date', 'дата', 'datasales'])),
                    'price': st.selectbox("Цена:", columns,
                                        index=auto_detect_column(columns, ['price', 'цена', 'стоимость'])),
                    'qty': st.selectbox("Количество:", columns,
                                      index=auto_detect_column(columns, ['qty', 'количество', 'кол'])),
                    'describe': st.selectbox("Описание (опционально):", 
                                           ["Не использовать"] + columns,
                                           index=0)
                }
                
                if columns_mapping['describe'] == "Не использовать":
                    columns_mapping['describe'] = None
    
    # Основная область
    if uploaded_file and df is not None:
        # Создание профилей магазинов
        with st.spinner("🔄 Анализ магазинов..."):
            store_profiles = create_store_profiles(df, columns_mapping['magazin'], 
                                                 columns_mapping['price'], columns_mapping['date'], 
                                                 columns_mapping['qty'])
            
            # Подготовка данных для обучения
            train_data = prepare_training_data(df, columns_mapping, store_profiles)
            
            # Определение признаков для модели
            feature_cols = ['gender', 'material', 'shape', 'is_polarized', 'has_uv', 
                          'store_avg_price', 'store_volume', 'price_match', 'price_segment', 
                          columns_mapping['magazin']]
            cat_features = ['gender', 'material', 'shape', 'price_segment', columns_mapping['magazin']]
            
            # Обучение модели
            model = train_model(train_data, columns_mapping['qty'], feature_cols, cat_features)
        
        st.success("🎯 Модель успешно обучена!")
        
        # Интерфейс ввода новой модели
        st.header("🆕 Характеристики новой модели")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_price = st.number_input("💰 Цена:", min_value=0, step=100, value=3000)
            new_gender = st.selectbox("👤 Пол:", ["Мужские", "Женские", "Унисекс"])
            new_material = st.selectbox("🔧 Материал:", ["Металл", "Пластик", "Дерево"])
            new_shape = st.selectbox("🕶️ Форма:", ["Авиатор", "Вайфарер", "Круглые", "Прямоугольные", "Кошачий глаз"])
        
        with col2:
            new_polarized = st.checkbox("⚡ Поляризация")
            new_uv = st.checkbox("🛡️ UV защита")
            
            # Определение ценового сегмента
            if new_price <= 2000:
                price_seg = "Эконом"
            elif new_price <= 5000:
                price_seg = "Средний"
            else:
                price_seg = "Премиум"
            st.info(f"Ценовой сегмент: {price_seg}")
        
        # Кнопка прогнозирования
        if st.button("🎯 ПОДОБРАТЬ МАГАЗИНЫ", type="primary"):
            # Подготовка данных новой модели
            new_product = {
                'price': new_price,
                'gender': new_gender,
                'material': new_material,
                'shape': new_shape,
                'is_polarized': int(new_polarized),
                'has_uv': int(new_uv),
                'price_segment': price_seg
            }
            
            # Получение прогнозов
            stores = df[columns_mapping['magazin']].unique()
            predictions = predict_for_stores(model, new_product, stores, store_profiles, feature_cols)
            
            # Сортировка результатов
            sorted_stores = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Отображение результатов
            st.header("🏆 Рекомендации магазинов")
            
            # ТОП-10
            st.subheader("✅ Лучшие магазины:")
            for i, (store, pred) in enumerate(sorted_stores[:10]):
                profile = store_profiles.get(store, {})
                compatibility = max(0, 100 - abs(new_price - profile.get('avg_price', new_price)) / new_price * 100)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"#{i+1} {store}", f"{pred:.0f} шт/мес")
                with col2:
                    st.metric("Совместимость", f"{compatibility:.0f}%")
                with col3:
                    st.metric("Средний чек", f"{profile.get('avg_price', 0):.0f} ₽")
            
            # Анти-рекомендации
            st.subheader("❌ Не рекомендуется:")
            for store, pred in sorted_stores[-3:]:
                st.write(f"• {store}: {pred:.0f} шт/мес (низкая совместимость)")
            
            # График
            fig = px.bar(x=[s[0] for s in sorted_stores[:15]], 
                        y=[s[1] for s in sorted_stores[:15]],
                        title="Прогноз продаж по магазинам",
                        labels={'x': 'Магазины', 'y': 'Прогноз (шт/мес)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Загрузите файл с данными для начала работы")
        
        # Демо данные
        st.header("📊 Пример структуры данных")
        demo_data = pd.DataFrame({
            'Magazin': ['Оптика Люкс', 'Стиль Центр', 'Модные Очки'],
            'Datasales': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Art': ['RB001', 'OAK002', 'GUC003'],
            'Price': [15000, 8000, 25000],
            'Qty': [2, 5, 1],
            'Describe': ['Ray-Ban авиаторы мужские металл поляризация', 
                        'Oakley спортивные унисекс пластик UV400',
                        'Gucci женские кошачий глаз премиум']
        })
        st.dataframe(demo_data)

if __name__ == "__main__":
    main()
