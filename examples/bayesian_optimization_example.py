"""
Example of HyperPhoenixCV with Bayesian optimization for smart hyperparameter tuning.

This example demonstrates:
- Using Bayesian optimization to prioritize promising parameters
- How the algorithm learns from previous iterations
- Comparison with random search and full grid search
- Getting results from Bayesian-optimized search
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hyperphoenixcv import HyperPhoenixCV
from sklearn.ensemble import RandomForestRegressor

# Load dataset
print("Загрузка данных...")
categories = ['alt.atheism', 'sci.space', 'comp.graphics', 'rec.sport.baseball']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X, y = newsgroups_train.data, newsgroups_train.target

# Create a pipeline
print("Создание пайплайна...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define parameter grid
param_grid = {
    'tfidf__max_features': [500, 1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2']
}

# Custom Bayesian optimizer with more trees for better predictions
custom_optimizer = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

# Create HyperPhoenixCV with Bayesian optimization
print("\nНастройка HyperPhoenixCV с байесовской оптимизацией...")
hp_bayesian = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    use_bayesian_optimization=True,
    bayesian_optimizer=custom_optimizer,
    checkpoint_path="bayesian_checkpoint.pkl",
    results_csv="bayesian_results.csv",
    verbose=True
)

# Run Bayesian-optimized hyperparameter search
print("\n" + "="*60)
print("ЗАПУСК ПОИСКА С БАЙЕСОВСКОЙ ОПТИМИЗАЦИЕЙ")
print("Байесовская оптимизация анализирует предыдущие результаты")
print("и предсказывает, какие параметры могут дать лучшие результаты")
print("="*60)
hp_bayesian.fit(X, y)

# Get results
bayesian_best_score = hp_bayesian.best_score_
bayesian_top_results = hp_bayesian.get_top_results(5)

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ")
print("="*50)
print(f"Лучший f1_macro score: {bayesian_best_score:.4f}")
print("\nТоп-5 комбинаций параметров:")
print(bayesian_top_results[['tfidf__max_features', 'tfidf__ngram_range', 
                           'clf__C', 'clf__penalty', 'mean_test_f1_macro']])

# For comparison, let's run random search with the same number of iterations
print("\n" + "="*50)
print("ЗАПУСК СЛУЧАЙНОГО ПОИСКА ДЛЯ СРАВНЕНИЯ")
print(f"Будет выполнено {len(hp_bayesian.cv_results_['params'])} итераций")
print("="*50)

hp_random = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    random_search=True,
    n_iter=len(hp_bayesian.cv_results_['params']),  # Same number of iterations
    random_state=42,
    verbose=True
)
hp_random.fit(X, y)

# Compare results
random_best_score = hp_random.best_score_

print("\n" + "="*50)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*50)
print(f"Байесовская оптимизация: {bayesian_best_score:.4f}")
print(f"Случайный поиск: {random_best_score:.4f}")

if bayesian_best_score > random_best_score:
    print("✅ Байесовская оптимизация превзошла случайный поиск!")
    print(f"  Улучшение: {(bayesian_best_score - random_best_score) * 100:.2f} процентных пунктов")
else:
    print("⚠️ Случайный поиск оказался лучше в этом запуске")
    print("  Это может происходить на ранних этапах оптимизации")

# Visualize the optimization process
print("\nСоздание графика прогресса оптимизации...")
try:
    # Get scores in order of evaluation
    bayesian_scores = [r[f'mean_test_f1_macro'] for r in hp_bayesian.cv_results_['params']]
    random_scores = [r[f'mean_test_f1_macro'] for r in hp_random.cv_results_['params']]
    
    # Calculate cumulative best score at each step
    bayesian_cummax = np.maximum.accumulate(bayesian_scores)
    random_cummax = np.maximum.accumulate(random_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bayesian_cummax, 'b-', label='Байесовская оптимизация', linewidth=2)
    plt.plot(random_cummax, 'r--', label='Случайный поиск', linewidth=2)
    
    plt.xlabel('Количество оцененных комбинаций')
    plt.ylabel('Лучший F1-макро скор')
    plt.title('Прогресс поиска гиперпараметров')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
    print("График сохранен как 'optimization_progress.png'")
    
    # Show plot in notebook environment (optional)
    try:
        import IPython
        if IPython.get_ipython():
            plt.show()
    except:
        pass
        
except Exception as e:
    print(f"⚠️ Не удалось создать график: {e}")

# Insights and recommendations
print("\n" + "="*50)
print("ИНСАЙТЫ И РЕКОМЕНДАЦИИ")
print("="*50)
print("Как работает байесовская оптимизация в HyperPhoenixCV:")
print("1. На первых итерациях исследует пространство параметров")
print("2. По мере накопления данных строит модель зависимости параметров от метрики")
print("3. Использует эту модель для выбора наиболее перспективных параметров")
print("4. Со временем фокусируется на самых многообещающих областях пространства")

print("\nРекомендации по использованию:")
print("- Используйте байесовскую оптимизацию, когда пространство параметров велико")
print("- Для небольших пространств параметров может быть достаточно полного перебора")
print("- Сочетайте с чекпоинтами для продолжения поиска после прерываний")
print("- Настройте bayesian_optimizer под свои задачи (количество деревьев и т.д.)")

# Clean up checkpoints
hp_bayesian.clear_checkpoint()
print("\nЧекпоинт байесовской оптимизации успешно удален.")

# Tip for users
print("\nСовет: Байесовская оптимизация особенно эффективна, когда оценка одной")
print("комбинации параметров занимает много времени (например, обучение глубоких моделей).")
print("В таких случаях экономия даже нескольких итераций может сэкономить часы вычислений!")
