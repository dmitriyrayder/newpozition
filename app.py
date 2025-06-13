import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def load_custom_css():
    """Загрузка кастомных стилей"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Настройка страницы
st.set_page_config(
    page_title="👓 Прогноз продаж очков",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка стилей
load_custom_css()

class GlassesSalesPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.categorical_features = []
        
    def extract_features_from_description(self, description):
        """Извлечение признаков из описания товара"""
        if pd.isna(description):
            description = ""
        
        desc_lower = description.lower()
        
        # Материал оправы
        frame_material = 'Другое'
        if any(word in desc_lower for word in ['металл', 'метал', 'steel', 'titanium']):
            frame_material = 'Металл'
        elif any(word in desc_lower for word in ['пластик', 'plastic', 'acetate']):
            frame_material = 'Пластик'
        elif any(word in desc_lower for word in ['дерев', 'wood', 'bamboo']):
            frame_material = 'Дерево'
            
        # Форма оправы
        frame_shape = 'Другое'
        if any(word in desc_lower for word in ['авиатор', 'aviator', 'пилот']):
            frame_shape = 'Авиатор'
        elif any(word in desc_lower for word in ['вайфарер', 'wayfarer']):
            frame_shape = 'Вайфарер'
        elif any(word in desc_lower for word in ['кошач', 'cat eye']):
            frame_shape = 'Кошачий глаз'
        elif any(word in desc_lower for word in ['круг', 'round']):
            frame_shape = 'Круглые'
        elif any(word in desc_lower for word in ['прямоуг', 'rectangle', 'квадрат']):
            frame_shape = 'Прямоугольные'
        elif any(word in desc_lower for word in ['спорт', 'sport']):
            frame_shape = 'Спортивные'
            
        # Цвет линз
        lens_color = 'Другое'
        if any(word in desc_lower for word in ['черн', 'black']):
            lens_color = 'Черный'
        elif any(word in desc_lower for word in ['коричн', 'brown']):
            lens_color = 'Коричневый'
        elif any(word in desc_lower for word in ['зеркал', 'mirror', 'серебр']):
            lens_color = 'Зеркальный'
        elif any(word in desc_lower for word in ['градиент', 'gradient']):
            lens_color = 'Градиентный'
        elif any(word in desc_lower for word in ['син', 'blue']):
            lens_color = 'Синий'
        elif any(word in desc_lower for word in ['зелен', 'green']):
            lens_color = 'Зеленый'
            
        # Пол
        gender = 'Унисекс'
        if any(word in desc_lower for word in ['мужск', 'men', 'male']):
            gender = 'Мужские'
        elif any(word in desc_lower for word in ['женск', 'women', 'female']):
            gender = 'Женские'
            
        # Поляризация и UV защита
        is_polarized = 1 if any(word in desc_lower for word in ['поляр', 'polar']) else 0
        has_uv_protection = 1 if any(word in desc_lower for word in ['uv', 'ультрафиолет']) else 0
        
        return {
            'frame_material': frame_material,
            'frame_shape': frame_shape,
            'lens_color': lens_color,
            'gender': gender,
            'is_polarized': is_polarized,
            'has_uv_protection': has_uv_protection
        }
    
    def create_price_segment(self, price):
        """Создание ценового сегмента"""
        if price < 2000:
            return 'Эконом'
        elif price < 5000:
            return 'Средний'
        else:
            return 'Премиум'
    
    def get_season(self, date):
        """Определение сезона по дате"""
        month = date.month
        if month in [12, 1, 2]:
            return 'Зима'
        elif month in [3, 4, 5]:
            return 'Весна'
        elif month in [6, 7, 8]:
            return 'Лето'
        else:
            return 'Осень'
    
    def prepare_training_data(self, df):
        """Подготовка данных для обучения"""
        try:
            # Преобразование даты
            df['Datasales'] = pd.to_datetime(df['Datasales'])
            
            # Извлечение признаков из описания
            features_from_desc = df['Describe'].apply(self.extract_features_from_description)
            features_df = pd.DataFrame(list(features_from_desc))
            
            # Объединение с основными данными
            df = pd.concat([df, features_df], axis=1)
            
            # Добавление временных признаков
            df['month'] = df['Datasales'].dt.month
            df['season'] = df['Datasales'].apply(self.get_season)
            df['day_of_week'] = df['Datasales'].dt.dayofweek
            
            # Добавление ценовых признаков
            df['price_segment'] = df['Price'].apply(self.create_price_segment)
            
            # Агрегация по артикулам
            agg_data = []
            
            for art in df['Art'].unique():
                art_data = df[df['Art'] == art].sort_values('Datasales')
                
                if len(art_data) == 0:
                    continue
                    
                # Первая дата продажи
                launch_date = art_data['Datasales'].min()
                
                # Продажи за первые 30 дней
                end_date = launch_date + timedelta(days=30)
                sales_30_days = art_data[
                    (art_data['Datasales'] >= launch_date) & 
                    (art_data['Datasales'] <= end_date)
                ]['Qty'].sum()
                
                # Характеристики из первой записи
                first_record = art_data.iloc[0]
                
                agg_data.append({
                    'Art': art,
                    'Magazin': first_record['Magazin'],
                    'Model': first_record['Model'],
                    'Segment': first_record['Segment'],
                    'Price': first_record['Price'],
                    'frame_material': first_record['frame_material'],
                    'frame_shape': first_record['frame_shape'],
                    'lens_color': first_record['lens_color'],
                    'gender': first_record['gender'],
                    'is_polarized': first_record['is_polarized'],
                    'has_uv_protection': first_record['has_uv_protection'],
                    'price_segment': first_record['price_segment'],
                    'launch_season': self.get_season(launch_date),
                    'launch_month': launch_date.month,
                    'sales_30_days': sales_30_days
                })
            
            return pd.DataFrame(agg_data)
        except Exception as e:
            st.error(f"Ошибка при подготовке данных: {str(e)}")
            return pd.DataFrame()
    
    def train_model(self, df):
        """Обучение модели"""
        try:
            # Подготовка признаков
            self.feature_columns = [
                'Price', 'frame_material', 'frame_shape', 'lens_color', 
                'gender', 'is_polarized', 'has_uv_protection', 'price_segment',
                'launch_season', 'launch_month', 'Segment'
            ]
            
            self.categorical_features = [
                'frame_material', 'frame_shape', 'lens_color', 
                'gender', 'price_segment', 'launch_season', 'Segment'
            ]
            
            X = df[self.feature_columns]
            y = df['sales_30_days']
            
            # Обучение CatBoost
            self.model = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                cat_features=self.categorical_features,
                verbose=False,
                random_seed=42
            )
            
            self.model.fit(X, y)
            return self.model
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {str(e)}")
            return None
    
    def predict(self, features_dict):
        """Прогнозирование для новой модели"""
        if self.model is None:
            return None
        
        try:
            # Создание DataFrame из словаря
            df = pd.DataFrame([features_dict])
            
            # Проверка наличия всех нужных колонок
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0 if col in ['is_polarized', 'has_uv_protection'] else 'Другое'
            
            X = df[self.feature_columns]
            prediction = self.model.predict(X)[0]
            
            return max(0, int(prediction))
        except Exception as e:
            st.error(f"Ошибка при прогнозировании: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Получение важности признаков"""
        if self.model is None:
            return None
        
        try:
            importance = self.model.get_feature_importance()
            feature_names = self.feature_columns
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        except Exception as e:
            st.error(f"Ошибка при получении важности признаков: {str(e)}")
            return None

def create_metric_card(title, value, delta=None):
    """Создание карточки с метрикой"""
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(title, value, delta)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Заголовок
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #667eea; font-size: 3rem; margin-bottom: 0;'>👓 Прогноз продаж очков</h1>
        <p style='color: #764ba2; font-size: 1.2rem;'>Умная аналитика для успешного бизнеса</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Инициализация предиктора
    if 'predictor' not in st.session_state:
        st.session_state.predictor = GlassesSalesPredictor()
    
    # Боковая панель
    with st.sidebar:
        st.markdown("### 📊 Навигация")
        tab = st.selectbox(
            "Выберите раздел:",
            ["📈 Обучение модели", "🔮 Прогнозирование", "📋 Анализ важности"]
        )
    
    if tab == "📈 Обучение модели":
        st.markdown("### 📤 Загрузка данных")
        
        # Секция загрузки
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**Загрузите CSV файл с историческими данными о продажах**")
        uploaded_file = st.file_uploader(
            "Перетащите файл сюда или нажмите для выбора",
            type=['csv'],
            help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            try:
                # Загрузка данных
                df = pd.read_csv(uploaded_file)
                
                # Проверка колонок
                required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Отсутствуют колонки: {', '.join(missing_cols)}")
                    return
                
                st.success(f"✅ Данные загружены! Строк: {len(df)}")
                
                # Предварительный просмотр
                with st.expander("👀 Предварительный просмотр данных"):
                    st.dataframe(df.head())
                
                # Обучение модели
                if st.button("🚀 Обучить модель", type="primary"):
                    with st.spinner("🔄 Обрабатываем данные и обучаем модель..."):
                        # Подготовка данных
                        training_data = st.session_state.predictor.prepare_training_data(df)
                        
                        if len(training_data) == 0:
                            st.error("❌ Не удалось подготовить данные для обучения")
                            return
                        
                        # Обучение
                        model = st.session_state.predictor.train_model(training_data)
                        
                        if model is not None:
                            st.session_state.model_trained = True
                            st.success("🎉 Модель успешно обучена!")
                            
                            # Статистика
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                create_metric_card("Моделей в обучении", len(training_data))
                            
                            with col2:
                                avg_sales = training_data['sales_30_days'].mean()
                                create_metric_card("Средние продажи за 30 дней", f"{avg_sales:.0f}")
                            
                            with col3:
                                max_sales = training_data['sales_30_days'].max()
                                create_metric_card("Максимальные продажи", f"{max_sales:.0f}")
                
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
    
    elif tab == "🔮 Прогнозирование":
        if not hasattr(st.session_state, 'model_trained') or not st.session_state.model_trained:
            st.warning("⚠️ Сначала обучите модель в разделе 'Обучение модели'")
            return
        
        st.markdown("### 🔮 Прогноз продаж новой модели")
        
        # Форма для ввода параметров
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💰 Коммерческие параметры**")
            price = st.number_input("Цена (руб.)", min_value=500, max_value=50000, value=3000, step=100)
            segment = st.selectbox("Сегмент", ["Солнцезащитные", "Оптические", "Спортивные"])
            
            st.markdown("**🎨 Дизайн оправы**")
            frame_material = st.selectbox("Материал оправы", ["Металл", "Пластик", "Дерево", "Другое"])
            frame_shape = st.selectbox("Форма оправы", ["Авиатор", "Вайфарер", "Кошачий глаз", "Круглые", "Прямоугольные", "Спортивные", "Другое"])
        
        with col2:
            st.markdown("**👁️ Параметры линз**")
            lens_color = st.selectbox("Цвет линз", ["Черный", "Коричневый", "Зеркальный", "Градиентный", "Синий", "Зеленый", "Другое"])
            is_polarized = st.checkbox("🌟 Поляризационные линзы")
            has_uv_protection = st.checkbox("☀️ UV защита")
            
            st.markdown("**👤 Целевая аудитория**")
            gender = st.selectbox("Пол", ["Мужские", "Женские", "Унисекс"])
            launch_month = st.selectbox("Месяц запуска", list(range(1, 13)), format_func=lambda x: f"{x} месяц")
        
        # Прогнозирование
        if st.button("🔍 Получить прогноз", type="primary"):
            # Подготовка данных
            features = {
                'Price': price,
                'frame_material': frame_material,
                'frame_shape': frame_shape,
                'lens_color': lens_color,
                'gender': gender,
                'is_polarized': 1 if is_polarized else 0,
                'has_uv_protection': 1 if has_uv_protection else 0,
                'price_segment': st.session_state.predictor.create_price_segment(price),
                'launch_season': st.session_state.predictor.get_season(datetime(2023, launch_month, 1)),
                'launch_month': launch_month,
                'Segment': segment
            }
            
            # Получение прогноза
            prediction = st.session_state.predictor.predict(features)
            
            if prediction is not None:
                # Результат
                st.markdown("### 📊 Результат прогноза")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    create_metric_card("Прогноз продаж за 30 дней", f"{prediction} шт.")
                
                with col2:
                    revenue = prediction * price
                    create_metric_card("Ожидаемая выручка", f"{revenue:,.0f} руб.")
                
                with col3:
                    if prediction < 30:
                        recommendation = "🔴 Низкий спрос"
                    elif prediction < 100:
                        recommendation = "🟡 Средний спрос"
                    else:
                        recommendation = "🟢 Высокий спрос"
                    create_metric_card("Рекомендация", recommendation)
                
                # Дополнительные рекомендации
                st.markdown("### 💡 Рекомендации по закупке")
                if prediction < 30:
                    st.info("📉 Модель может показать слабые продажи. Рекомендуется ограниченная тестовая закупка.")
                elif prediction < 100:
                    st.success("📈 Модель имеет средний потенциал. Рекомендуется стандартная закупка.")
                else:
                    st.success("🚀 Модель имеет высокий потенциал! Рекомендуется увеличенная закупка.")
    
    elif tab == "📋 Анализ важности":
        if not hasattr(st.session_state, 'model_trained') or not st.session_state.model_trained:
            st.warning("⚠️ Сначала обучите модель в разделе 'Обучение модели'")
            return
        
        st.markdown("### 📊 Важность признаков")
        
        # Получение важности признаков
        importance_df = st.session_state.predictor.get_feature_importance()
        
        if importance_df is not None:
            # График важности
            fig = px.bar(
                importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Топ-10 наиболее важных признаков",
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица важности
            st.markdown("### 📋 Детальная таблица важности")
            st.dataframe(
                importance_df.style.format({'importance': '{:.2f}'}),
                use_container_width=True
            )
    
    # Футер
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "💜 Создано с любовью для успешного бизнеса"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
