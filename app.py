import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Конфигурация и стили
st.set_page_config(page_title="💖 Модный Советник по Продажам", page_icon="🕶️", layout="wide")
st.markdown("""<style>
.main-header {font-size: 2.5rem; color: #e74c3c; text-align: center; margin-bottom: 2rem; font-weight: bold;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💖 Модный Советник по Продажам</h1>', unsafe_allow_html=True)

# Вспомогательные функции
@st.cache_data
def auto_detect_column(columns, keywords):
    for keyword in keywords:
        for i, col in enumerate(columns):
            if keyword.lower() in col.lower():
                return i
    return 0

def create_synthetic_features(df, feature_configs):
    """Создание синтетических признаков для демонстрации"""
    np.random.seed(42)
    feature_options = {
        'gender': ['Мужские', 'Женские', 'Унисекс'],
        'material': ['Металл', 'Пластик', 'Дерево', 'Комбинированный'],
        'shape': ['Авиатор', 'Вайфарер', 'Круглые', 'Прямоугольные', 'Кошачий глаз', 'Спортивные'],
        'brand': ['Ray-Ban', 'Oakley', 'Gucci', 'Prada', 'Другой']
    }
    
    for feature, config in feature_configs.items():
        if config['source'] == "Ввести вручную для новой модели" and feature not in df.columns:
            df[feature] = np.random.choice(feature_options.get(feature, ['Другое']), size=len(df))
        elif config['source'] == "Выбрать колонку" and 'column' in config:
            df[feature] = df[config['column']]
    
    return df

class RecommendationEngine:
    def __init__(self, df, new_features, feature_weights=None):
        self.df = df
        self.new_features = new_features
        self.weights = feature_weights or {'price': 0.30, 'gender': 0.25, 'material': 0.25, 'shape': 0.20}
        self.stores = df['store'].unique()
    
    def calculate_compatibility(self, store_data):
        """Расчет совместимости по всем признакам"""
        scores = {}
        
        # Совместимость по цене
        if not store_data.empty and 'price' in store_data.columns:
            avg_price = store_data['price'].mean()
            price_diff = abs(self.new_features['price'] - avg_price) / max(avg_price, 1)
            scores['price'] = max(0.2, 1 - min(price_diff, 1.0))
        else:
            scores['price'] = 0.5
        
        # Совместимость по категориальным признакам
        for feature in ['gender', 'material', 'shape', 'brand']:
            if feature in self.new_features and feature in store_data.columns:
                feature_counts = store_data[feature].value_counts()
                if self.new_features[feature] in feature_counts.index:
                    share = feature_counts[self.new_features[feature]] / len(store_data)
                    scores[feature] = min(1.0, share * 2)
                else:
                    scores[feature] = 0.3
            else:
                scores[feature] = 0.5
        
        return scores
    
    def predict_sales(self, store_data, compatibility_scores):
        """Прогноз продаж на основе сегментного анализа"""
        if store_data.empty:
            return 10
        
        # Фильтрация похожих товаров
        similar_items = store_data.copy()
        price_range = self.new_features['price'] * 0.3
        similar_items = similar_items[
            (similar_items['price'] >= self.new_features['price'] - price_range) &
            (similar_items['price'] <= self.new_features['price'] + price_range)
        ]
        
        # Фильтрация по признакам
        for feature in ['gender', 'material', 'shape']:
            if feature in self.new_features and feature in similar_items.columns:
                similar_items = similar_items[
                    (similar_items[feature] == self.new_features[feature]) |
                    (similar_items[feature] == 'Унисекс') if feature == 'gender' else
                    (similar_items[feature] == self.new_features[feature])
                ]
        
        # Расчет прогноза
        if not similar_items.empty:
            segment_avg = similar_items['quantity'].mean()
            segment_multiplier = min(2.0, len(similar_items['article'].unique()) / 5)
            predicted = segment_avg * segment_multiplier
        else:
            unique_articles = store_data['article'].nunique()
            predicted = store_data['quantity'].sum() / max(1, unique_articles)
        
        # Применение совместимости
        overall_compatibility = sum(compatibility_scores[k] * self.weights.get(k, 0) 
                                  for k in compatibility_scores.keys())
        
        return max(5, predicted * overall_compatibility)
    
    def generate_recommendations(self):
        """Генерация рекомендаций для всех магазинов"""
        recommendations = []
        
        for store in self.stores:
            store_data = self.df[self.df['store'] == store]
            compatibility_scores = self.calculate_compatibility(store_data)
            predicted_sales = self.predict_sales(store_data, compatibility_scores)
            
            overall_compatibility = sum(compatibility_scores[k] * self.weights.get(k, 0) 
                                      for k in compatibility_scores.keys())
            
            recommendations.append({
                'store': store,
                'predicted_sales': predicted_sales,
                'compatibility': overall_compatibility,
                'scores': compatibility_scores,
                'avg_price': store_data['price'].mean() if not store_data.empty else self.new_features['price'],
                'total_items': len(store_data),
                'unique_articles': store_data['article'].nunique() if not store_data.empty else 0
            })
        
        return sorted(recommendations, key=lambda x: x['predicted_sales'], reverse=True)

def display_recommendations(recommendations, new_features):
    """Отображение рекомендаций с оптимизированным интерфейсом"""
    st.subheader("🏆 Рекомендуемые магазины")
    
    # Топ-10 рекомендаций
    top_recs = recommendations[:min(10, len(recommendations))]
    
    for i, rec in enumerate(top_recs):
        # Определение статуса
        if rec['compatibility'] >= 0.8:
            status, color = "🟢 Отлично", "success"
        elif rec['compatibility'] >= 0.6:
            status, color = "🟡 Хорошо", "warning"
        else:
            status, color = "🔴 Удовлетворительно", "error"
        
        with st.expander(f"#{i+1} {rec['store']} - {status} - Прогноз: {rec['predicted_sales']:.0f} шт/месяц"):
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("📈 Прогноз продаж", f"{rec['predicted_sales']:.0f} шт/мес")
            with col2: st.metric("🎯 Совместимость", f"{rec['compatibility']:.1%}")
            with col3: st.metric("💰 Средняя цена", f"{rec['avg_price']:.0f} ₽")
            with col4: st.metric("📦 Товаров", f"{rec['unique_articles']}")
            
            # Детализация совместимости только по выбранным признакам
            criteria_map = {'price': '💰 Цена', 'gender': '👤 Пол', 'material': '🔧 Материал', 
                          'shape': '🕶️ Форма', 'brand': '🏷️ Бренд'}
            
            compatibility_data = []
            for criterion, score in rec['scores'].items():
                if criterion in new_features:  # Показываем только выбранные признаки
                    compatibility_data.append({
                        'Критерий': criteria_map.get(criterion, criterion),
                        'Совместимость': f"{score:.1%}",
                        'Оценка': "Отлично" if score >= 0.8 else "Хорошо" if score >= 0.6 else "Слабо"
                    })
            
            if compatibility_data:
                st.dataframe(pd.DataFrame(compatibility_data), use_container_width=True, hide_index=True)
            
            # Причины рекомендации на основе выбранных признаков
            reasons = []
            for criterion, score in rec['scores'].items():
                if criterion in new_features and score > 0.7:
                    if criterion == 'price':
                        reasons.append("✅ Отличная совместимость по цене")
                    elif criterion == 'gender':
                        reasons.append(f"✅ Высокий спрос на {new_features['gender'].lower()} товары")
                    elif criterion == 'material':
                        reasons.append(f"✅ Популярный материал: {new_features['material']}")
                    elif criterion == 'shape':
                        reasons.append(f"✅ Востребованная форма: {new_features['shape']}")
                    elif criterion == 'brand':
                        reasons.append(f"✅ Известный бренд: {new_features['brand']}")
            
            if rec['unique_articles'] > 50:
                reasons.append("✅ Большой ассортимент")
            
            if reasons:
                st.write("**Преимущества:**")
                for reason in reasons[:4]:  # Ограничиваем количество
                    st.write(reason)

# Основной интерфейс
with st.sidebar:
    st.header("🔧 Настройки")
    uploaded_file = st.file_uploader("Загрузите файл с данными", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        # Загрузка данных
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"📊 Данные загружены: {len(df)} строк, {len(df.columns)} колонок")
        
        # Настройка колонок
        st.subheader("🎯 Настройка колонок")
        cols = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            col_store = st.selectbox("Магазин:", cols, index=auto_detect_column(cols, ['magazin', 'магазин', 'store']))
            col_date = st.selectbox("Дата:", cols, index=auto_detect_column(cols, ['date', 'дата']))
            col_article = st.selectbox("Артикул:", cols, index=auto_detect_column(cols, ['art', 'артикул', 'sku']))
        with col2:
            col_price = st.selectbox("Цена:", cols, index=auto_detect_column(cols, ['price', 'цена']))
            col_qty = st.selectbox("Количество:", cols, index=auto_detect_column(cols, ['qty', 'количество']))
        
        # Создание рабочего датасета
        analysis_df = df.rename(columns={
            col_store: 'store', col_date: 'date', col_article: 'article',
            col_price: 'price', col_qty: 'quantity'
        })
        
        # Настройка признаков
        st.subheader("🎨 Характеристики товара")
        col1, col2 = st.columns(2)
        
        feature_configs = {}
        with col1:
            feature_configs['gender'] = {'source': st.radio("👤 Пол:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
            feature_configs['material'] = {'source': st.radio("🔧 Материал:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
        with col2:
            feature_configs['shape'] = {'source': st.radio("🕶️ Форма:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
            feature_configs['brand'] = {'source': st.radio("🏷️ Бренд:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
        
        # Создание признаков
        analysis_df = create_synthetic_features(analysis_df, feature_configs)
        
        # Ввод новой модели
        st.subheader("🆕 Новая модель")
        col1, col2 = st.columns(2)
        
        new_features = {}
        with col1:
            new_features['price'] = st.number_input("💰 Цена:", min_value=0, step=100, value=5000)
            new_features['gender'] = st.selectbox("👤 Пол:", ["Мужские", "Женские", "Унисекс"])
            new_features['material'] = st.selectbox("🔧 Материал:", ["Металл", "Пластик", "Дерево", "Комбинированный"])
        with col2:
            new_features['shape'] = st.selectbox("🕶️ Форма:", ["Авиатор", "Вайфарер", "Круглые", "Прямоугольные", "Кошачий глаз", "Спортивные"])
            new_features['brand'] = st.selectbox("🏷️ Бренд:", ["Ray-Ban", "Oakley", "Gucci", "Prada", "Другой"])
        
        # Генерация рекомендаций
        if st.button("🎯 ПОДОБРАТЬ МАГАЗИНЫ", type="primary"):
            with st.spinner("Анализ данных..."):
                engine = RecommendationEngine(analysis_df, new_features)
                recommendations = engine.generate_recommendations()
                display_recommendations(recommendations, new_features)
                
                # Общая статистика
                st.subheader("📊 Статистика")
                total_predicted = sum(r['predicted_sales'] for r in recommendations)
                avg_compatibility = np.mean([r['compatibility'] for r in recommendations])
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Общий прогноз", f"{total_predicted:.0f} шт/месяц")
                with col2: st.metric("Средняя совместимость", f"{avg_compatibility:.1%}")
                with col3: st.metric("Магазинов", len(recommendations))
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
else:
    st.info("👈 Загрузите файл для начала работы")
    st.subheader("📋 Инструкция:")
    steps = ["Загрузите CSV/Excel файл", "Настройте колонки данных", "Выберите источники признаков", 
             "Введите параметры новой модели", "Получите рекомендации"]
    for i, step in enumerate(steps, 1):
        st.write(f"{i}. {step}")
