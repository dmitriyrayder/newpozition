import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

# Отключаем логирование Optuna, чтобы не засорять консоль
optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(page_title="Рекомендатор магазинов", layout="wide")

st.title("🎯 Рекомендатор магазинов для нового товара")

# Загрузка файла
dataset_file = st.file_uploader("\U0001F4C2 Загрузите CSV-файл с данными о продажах", type=["csv"])

# --- Логика обучения модели и подготовки данных ---
@st.cache_data
def process_data_and_train(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Datasales'])
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return None, None, None, None, None

    # 1. ПРАВИЛЬНАЯ АГРЕГАЦИЯ ДАННЫХ
    st.info("Шаг 1: Подготовка данных. Агрегируем продажи за первые 30 дней...")
    
    df = df.dropna(subset=['Qty', 'Art', 'Magazin'])
    df = df.sort_values(by=['Art', 'Magazin', 'Datasales'])

    # Находим дату первой продажи для каждой пары (Товар, Магазин)
    first_sale_dates = df.groupby(['Art', 'Magazin'])['Datasales'].first().reset_index()
    first_sale_dates.rename(columns={'Datasales': 'first_sale_date'}, inplace=True)

    df = pd.merge(df, first_sale_dates, on=['Art', 'Magazin'])
    
    # Оставляем только продажи в течение 30 дней с первой продажи
    df_30_days = df[df['Datasales'] <= (df['first_sale_date'] + pd.Timedelta(days=30))].copy()

    # Агрегируем данные, чтобы получить одну строку на пару (Товар, Магазин)
    agg_logic = {
        'Qty': 'sum', 'Sum': 'sum', 'Price': 'mean',
        'Model': 'first', 'brand': 'first', 'Segment': 'first',
        'color': 'first', 'formaoprav': 'first', 'Sex': 'first',
        'Metal-Plastic': 'first'
    }
    df_agg = df_30_days.groupby(['Art', 'Magazin'], as_index=False).agg(agg_logic)
    df_agg.rename(columns={'Qty': 'Qty_30_days'}, inplace=True)
    
    if df_agg.empty:
        st.error("Не удалось сформировать агрегированные данные. Проверьте содержимое файла.")
        return None, None, None, None, None

    # 2. ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ И ЦЕЛИ
    target = 'Qty_30_days'
    # 'Art' и 'Model' убраны из признаков, так как это ID. 'Magazin' - ключевой признак!
    cat_features = ['Magazin', 'brand', 'Segment', 'color', 'formaoprav', 'Sex', 'Metal-Plastic']
    # Убираем лишние колонки
    drop_cols = ['Sum', 'Art', 'Model'] 
    
    # Очистка и подготовка
    df_agg = df_agg.drop(columns=drop_cols, errors='ignore')
    features = [col for col in df_agg.columns if col != target]
    
    # Убедимся, что все категориальные признаки имеют строковый тип
    for col in cat_features:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].astype(str)

    X = df_agg[features]
    y = df_agg[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 3. АВТОПОДБОР ПАРАМЕТРОВ с OPTUNA
    st.info("Шаг 2: Автоподбор параметров модели с помощью Optuna...")
    
    def objective(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': 0,
            'random_seed': 42
        }
        
        model = CatBoostRegressor(**params)
        # Обучаем на тренировочной и валидируемся на тестовой
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        return mae

    # Запускаем оптимизацию
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) # 30 попыток для скорости, можно увеличить до 50-100
    
    best_params = study.best_params
    st.success(f"Лучшие параметры найдены: {best_params}")

    # 4. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ
    st.info("Шаг 3: Обучение финальной модели на лучших параметрах...")
    final_model = CatBoostRegressor(**best_params, iterations=1500, verbose=0, random_seed=42)
    final_model.fit(X, y, cat_features=cat_features) # Обучение на всех данных

    # Оценка качества финальной модели на отложенной выборке
    test_preds = final_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, test_preds)
    final_r2 = r2_score(y_test, test_preds)

    metrics = {'MAE': final_mae, 'R2': final_r2}

    return final_model, df_agg, features, cat_features, metrics


# --- Основной блок Streamlit ---
if dataset_file:
    # Запускаем весь процесс
    model, df_agg, features, cat_features, metrics = process_data_and_train(dataset_file)

    if model:
        st.header("📊 Оценка качества модели")
        col1, col2 = st.columns(2)
        col1.metric("Средняя абсолютная ошибка (MAE)", f"{metrics['MAE']:.2f} шт.")
        col2.metric("Коэффициент детерминации (R²)", f"{metrics['R2']:.2%}")
        st.caption("MAE показывает, на сколько штук в среднем ошибается прогноз. R² показывает, какую долю дисперсии данных объясняет модель.")

        st.header("✍️ Введите характеристики нового товара")
        with st.form("product_form"):
            # Используем text_input для новых товаров, чтобы можно было вводить невиданные ранее значения
            col1, col2 = st.columns(2)
            with col1:
                brand = st.text_input("Brand (бренд)", help="Например, Ray-Ban")
                forma = st.text_input("Forma oprav (форма оправы)", help="Например, Авиатор")
                sex = st.selectbox("Sex (пол)", df_agg['Sex'].unique())
                price = st.number_input("Price (цена)", min_value=0.0, step=100.0, format="%.2f")
            
            with col2:
                segment = st.selectbox("Segment (сегмент)", df_agg['Segment'].unique())
                color = st.text_input("Color (цвет)", help="Например, Черный")
                material = st.selectbox("Metal-Plastic (материал)", df_agg['Metal-Plastic'].unique())
            
            submitted = st.form_submit_button("🚀 Получить рекомендации")

        if submitted:
            # Создаем DataFrame для предсказания
            magaziny = df_agg['Magazin'].unique()
            
            # Собираем данные нового товара в словарь
            new_product_data = {
                'brand': brand,
                'Segment': segment,
                'color': color,
                'formaoprav': forma,
                'Sex': sex,
                'Metal-Plastic': material,
                'Price': price
            }

            # Создаем строки для каждого магазина
            recs_list = []
            for magazin in magaziny:
                row = new_product_data.copy()
                row['Magazin'] = magazin
                recs_list.append(row)
            
            recs_df = pd.DataFrame(recs_list)
            # Упорядочиваем колонки как в обучающей выборке
            recs_df = recs_df[features]

            # Делаем предсказание
            recs_df['Pred_Qty_30_days'] = model.predict(recs_df).round(0)
            
            # Рассчитываем относительный рейтинг
            max_pred = recs_df['Pred_Qty_30_days'].max()
            if max_pred > 0:
                recs_df['Rating_%'] = (recs_df['Pred_Qty_30_days'] / max_pred * 100).round(0)
            else:
                recs_df['Rating_%'] = 0

            top_magaziny = recs_df.sort_values(by='Pred_Qty_30_days', ascending=False).reset_index(drop=True)
            
            # Убираем отрицательные прогнозы, если они есть
            top_magaziny['Pred_Qty_30_days'] = top_magaziny['Pred_Qty_30_days'].apply(lambda x: max(0, x))

            st.subheader("\U0001F4C8 Рекомендованные магазины для нового товара")
            st.table(top_magaziny[['Magazin', 'Pred_Qty_30_days', 'Rating_%']].rename(columns={
                'Magazin': 'Магазин',
                'Pred_Qty_30_days': 'Прогноз продаж (30 дней, шт.)',
                'Rating_%': 'Рейтинг (%)'
            }))
else:
    st.info("Пожалуйста, загрузите CSV-файл с данными о продажах для начала работы.")