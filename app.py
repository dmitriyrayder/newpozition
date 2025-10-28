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
.profile-card {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;}
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
        'brand': ['Ray-Ban', 'Oakley', 'Gucci', 'Prada', 'Другой'],
        'segment': ['Эконом', 'Средний', 'Премиум', 'Люкс']
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
        # ВИПРАВЛЕНО: нормалізація ваг до суми 1.0
        self.weights = feature_weights or {'price': 0.25, 'gender': 0.20, 'material': 0.20, 'shape': 0.20, 'segment': 0.15}
        # Перевірка та нормалізація ваг
        weights_sum = sum(self.weights.values())
        if abs(weights_sum - 1.0) > 0.01:  # Якщо сума не дорівнює 1.0
            self.weights = {k: v/weights_sum for k, v in self.weights.items()}
        self.stores = df['store'].unique()
    
    def calculate_compatibility(self, store_data):
        """Расчет совместимости по всем признакам"""
        scores = {}
        
        # Совместимость по цене
        if not store_data.empty and 'price' in store_data.columns:
            avg_price = store_data['price'].mean()
            # ВИПРАВЛЕНО: обробка випадку avg_price = 0
            if avg_price > 0:
                price_diff = abs(self.new_features['price'] - avg_price) / avg_price
                scores['price'] = max(0.2, 1 - min(price_diff, 1.0))
            else:
                scores['price'] = 0.5  # Нейтральна оцінка, якщо немає даних про ціни
        else:
            scores['price'] = 0.5
        
        # Совместимость по категориальным признакам
        for feature in ['gender', 'material', 'shape', 'brand', 'segment']:
            if feature in self.new_features and feature in store_data.columns:
                # Если выбрано "Учитывать все", ставим высокую совместимость
                if self.new_features[feature] == "Учитывать все":
                    scores[feature] = 0.9
                else:
                    feature_counts = store_data[feature].value_counts()
                    if self.new_features[feature] in feature_counts.index:
                        share = feature_counts[self.new_features[feature]] / len(store_data)
                        # ВИПРАВЛЕНО: більш логічна формула з обмеженням
                        scores[feature] = max(0.3, min(1.0, 0.5 + share * 1.5))
                    else:
                        scores[feature] = 0.3
            else:
                scores[feature] = 0.5
        
        return scores
    
    def calculate_profile_sales(self, store_data):
        """Расчет суммы продаж товаров с аналогичным профилем"""
        if store_data.empty:
            return 0, 0
        
        # Фильтрация товаров с похожим профилем
        similar_items = store_data.copy()
        
        # Фильтр по цене (±30% от целевой цены)
        price_range = self.new_features['price'] * 0.3
        similar_items = similar_items[
            (similar_items['price'] >= self.new_features['price'] - price_range) &
            (similar_items['price'] <= self.new_features['price'] + price_range)
        ]
        
        # Фильтр по категориальным признакам
        for feature in ['gender', 'material', 'shape', 'brand', 'segment']:
            if feature in self.new_features and feature in similar_items.columns:
                # Если выбрано "Учитывать все", не фильтруем по этому признаку
                if self.new_features[feature] == "Учитывать все":
                    continue
                elif feature == 'gender' and self.new_features[feature] == 'Унисекс':
                    # Для унисекс включаем все категории
                    continue
                elif feature == 'gender':
                    # Для мужских/женских включаем также унисекс
                    similar_items = similar_items[
                        (similar_items[feature] == self.new_features[feature]) |
                        (similar_items[feature] == 'Унисекс')
                    ]
                else:
                    similar_items = similar_items[similar_items[feature] == self.new_features[feature]]
        
        total_sales = similar_items['quantity'].sum() if not similar_items.empty else 0
        unique_articles = similar_items['article'].nunique() if not similar_items.empty else 0
        
        return total_sales, unique_articles
    
    def predict_sales(self, store_data, compatibility_scores):
        """Прогноз продаж на основе сегментного анализа"""
        if store_data.empty:
            return 10
        
        # Используем данные о похожих товарах
        profile_sales, profile_articles = self.calculate_profile_sales(store_data)
        
        if profile_sales > 0 and profile_articles > 0:
            # Если есть данные о похожих товарах, используем их
            predicted = profile_sales / profile_articles
        else:
            # Иначе используем общую статистику магазина
            unique_articles = store_data['article'].nunique()
            predicted = store_data['quantity'].sum() / max(1, unique_articles)
        
        # Применение совместимости
        overall_compatibility = sum(compatibility_scores[k] * self.weights.get(k, 0) 
                                  for k in compatibility_scores.keys())
        
        # ВИПРАВЛЕНО: адаптивний мінімум залежно від сумісності
        min_sales = 5 if overall_compatibility > 0.5 else 2
        return max(min_sales, predicted * overall_compatibility)
    
    def generate_recommendations(self):
        """Генерация рекомендаций для всех магазинов"""
        recommendations = []
        
        for store in self.stores:
            store_data = self.df[self.df['store'] == store]
            compatibility_scores = self.calculate_compatibility(store_data)
            predicted_sales = self.predict_sales(store_data, compatibility_scores)
            profile_sales, profile_articles = self.calculate_profile_sales(store_data)
            
            overall_compatibility = sum(compatibility_scores[k] * self.weights.get(k, 0) 
                                      for k in compatibility_scores.keys())
            
            recommendations.append({
                'store': store,
                'predicted_sales': predicted_sales,
                'compatibility': overall_compatibility,
                'scores': compatibility_scores,
                'profile_sales': profile_sales,
                'unique_articles': profile_articles
            })
        
        return sorted(recommendations, key=lambda x: x['predicted_sales'], reverse=True)

def display_recommendations(recommendations, new_features):
    """Вывод рекомендаций с детальным анализом"""
    if not recommendations:
        st.warning("Нет данных для отображения рекомендаций")
        return
    
    st.subheader(f"🎯 Топ-{min(10, len(recommendations))} магазинов для размещения")
    
    for i, rec in enumerate(recommendations[:10], 1):
        with st.expander(f"#{i} | {rec['store']} — Прогноз: {rec['predicted_sales']:.0f} шт/месяц | Совместимость: {rec['compatibility']:.1%}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 {rec['predicted_sales']:.0f}</h3>
                    <p>Прогноз продаж/месяц</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 {rec['compatibility']:.1%}</h3>
                    <p>Совместимость</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📦 {rec['profile_sales']:.0f}</h3>
                    <p>Продажи похожих</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Анализ причин рекомендации
            reasons = []
            warnings = []
            
            for criterion, score in rec['scores'].items():
                if criterion in new_features:
                    if score > 0.7:
                        if criterion == 'price':
                            reasons.append("✅ Отличная совместимость по цене с ассортиментом магазина")  # ВИПРАВЛЕНО
                        elif criterion == 'gender':
                            reasons.append(f"✅ Высокий спрос в магазине на {new_features['gender'].lower()} товары")
                        elif criterion == 'material':
                            reasons.append(f"✅ Популярный в магазине материал: {new_features['material']}")
                        elif criterion == 'shape':
                            reasons.append(f"✅ Востребованная в магазине форма: {new_features['shape']}")
                        elif criterion == 'brand':
                            reasons.append(f"✅ Успешно продающийся бренд: {new_features['brand']}")
                    elif score < 0.5:
                        if criterion == 'price':
                            warnings.append("⚠️ Цена не соответствует ценовому сегменту магазина")
                        elif criterion == 'gender':
                            warnings.append(f"⚠️ Низкий спрос на {new_features['gender'].lower()} товары")
                        elif criterion == 'material':
                            warnings.append(f"⚠️ Материал {new_features['material']} редко покупают в этом магазине")
                        elif criterion == 'shape':
                            warnings.append(f"⚠️ Форма {new_features['shape']} не популярна в магазине")
                        elif criterion == 'brand':
                            warnings.append(f"⚠️ Бренд {new_features['brand']} слабо представлен в магазине")
            
            if rec['unique_articles'] > 50:
                reasons.append("✅ Большой и разнообразный ассортимент")
            
            if reasons:
                st.write("**💪 Преимущества размещения:**")
                for reason in reasons[:4]:
                    st.write(reason)
            
            if warnings:
                st.write("**⚠️ Потенциальные риски:**")
                for warning in warnings[:3]:
                    st.write(warning)

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
        
        # Проверка наличия колонки store
        if 'store' not in analysis_df.columns:
            st.error("Не удалось найти колонку с магазинами. Проверьте настройки колонок.")
            st.stop()
        
        # Настройка признаков
        st.subheader("🎨 Характеристики товара")
        col1, col2 = st.columns(2)
        
        feature_configs = {}
        with col1:
            feature_configs['gender'] = {'source': st.radio("👤 Пол:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
            feature_configs['material'] = {'source': st.radio("🔧 Материал:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
            feature_configs['segment'] = {'source': st.radio("💎 Сегмент:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
        with col2:
            feature_configs['shape'] = {'source': st.radio("🕶️ Форма:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
            feature_configs['brand'] = {'source': st.radio("🏷️ Бренд:", ["Ввести вручную для новой модели", "Выбрать колонку", "Не использовать"])}
        
        # Создание признаков
        analysis_df = create_synthetic_features(analysis_df, feature_configs)
        
        # Получение уникальных значений из данных для выпадающих списков
        unique_values = {}
        for feature in ['gender', 'material', 'shape', 'brand', 'segment']:
            if feature in analysis_df.columns:
                unique_values[feature] = sorted(analysis_df[feature].dropna().unique().tolist())
        
        # Ввод новой модели
        st.subheader("🆕 Введите характеристики новой модели")
        col1, col2 = st.columns(2)
        
        new_features = {}
        with col1:
            new_features['price'] = st.number_input("💰 Цена:", min_value=0, step=100, value=5000)
            
            # Пол с опцией "Учитывать все"
            gender_options = ["Учитывать все"] + unique_values.get('gender', ["Мужские", "Женские", "Унисекс"])
            new_features['gender'] = st.selectbox("👤 Пол:", gender_options)
            
            # Материал с опцией "Учитывать все"
            material_options = ["Учитывать все"] + unique_values.get('material', ["Металл", "Пластик", "Дерево", "Комбинированный"])
            new_features['material'] = st.selectbox("🔧 Материал:", material_options)
            
        with col2:
            # Форма с опцией "Учитывать все"
            shape_options = ["Учитывать все"] + unique_values.get('shape', ["Авиатор", "Вайфарер", "Круглые", "Прямоугольные", "Кошачий глаз", "Спортивные"])
            new_features['shape'] = st.selectbox("🕶️ Форма:", shape_options)
            
            new_features['brand'] = st.selectbox("🏷️ Бренд:", unique_values.get('brand', ["Ray-Ban", "Oakley", "Gucci", "Prada", "Другой"]))
            
            # Сегмент с опцией "Учитывать все"
            segment_options = ["Учитывать все"] + unique_values.get('segment', ["Эконом", "Средний", "Премиум", "Люкс"])
            new_features['segment'] = st.selectbox("💎 Сегмент:", segment_options)
        
        # Генерация рекомендаций
        if st.button("🎯 ПОДОБРАТЬ МАГАЗИНЫ", type="primary"):
            with st.spinner("Анализ данных и поиск похожих товаров..."):
                engine = RecommendationEngine(analysis_df, new_features)
                recommendations = engine.generate_recommendations()
                display_recommendations(recommendations, new_features)
                
                # Общая статистика
                st.subheader("📊 Общая статистика по всем магазинам")
                total_predicted = sum(r['predicted_sales'] for r in recommendations)
                avg_compatibility = np.mean([r['compatibility'] for r in recommendations])
                total_profile_sales = sum(r['profile_sales'] for r in recommendations)
                stores_with_profile = sum(1 for r in recommendations if r['profile_sales'] > 0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Общий прогноз", f"{total_predicted:.0f} шт/месяц")
                with col2: st.metric("Средняя совместимость", f"{avg_compatibility:.1%}")
                with col3: st.metric("Продажи похожих товаров", f"{total_profile_sales:.0f} шт")
                with col4: st.metric("Магазинов с профилем", f"{stores_with_profile}/{len(recommendations)}")
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
else:
    st.info("👈 Загрузите файл для начала работы")
    st.subheader("📋 Инструкция:")
    steps = ["Загрузите CSV/Excel файл", "Настройте колонки данных", "Выберите источники признаков", 
             "Введите параметры новой модели", "Получите рекомендации с анализом похожих товаров"]
    for i, step in enumerate(steps, 1):
        st.write(f"{i}. {step}")
