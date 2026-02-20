"""
Example of HyperPhoenixCV with random search for efficient hyperparameter tuning.

This example demonstrates:
- Using random search instead of full grid search
- Setting n_iter to control the number of combinations
- How random search can find good parameters faster than full grid search
- Getting results from random search
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hyperphoenixcv import HyperPhoenixCV

# Load dataset
print("Загрузка данных...")
categories = ['alt.atheism', 'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X, y = newsgroups_train.data, newsgroups_train.target

# Create a pipeline
print("Создание пайплайна...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, solver='saga', penalty='l1'))
])

# Define a larger parameter grid to demonstrate the advantage of random search
param_grid = {
    'tfidf__max_features': [500, 1000, 5000, 10000, 15000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__use_idf': [True, False],
    'tfidf__smooth_idf': [True, False],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear', 'saga']
}

# Calculate total combinations to show the advantage of random search
total_combinations = 1
for v in param_grid.values():
    total_combinations *= len(v)
print(f"\nОбщее количество возможных комбинаций: {total_combinations}")
print("Полный перебор займет слишком много времени!")
print("Случайный поиск проверит только небольшую часть из них.\n")

# Create HyperPhoenixCV with random search
print("Настройка HyperPhoenixCV с случайным поиском...")
hp = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    random_search=True,  # Включаем случайный поиск
    n_iter=50,           # Количество случайных комбинаций для проверки
    random_state=42,     # Для воспроизводимости
    checkpoint_path="random_search_checkpoint.pkl",
    results_csv="random_search_results.csv",
    verbose=True
)

# Run hyperparameter search
print("\nЗапуск случайного поиска гиперпараметров...")
hp.fit(X, y)

# Print results
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ СЛУЧАЙНОГО ПОИСКА")
print("="*50)
print(f"Проверено {hp.n_iter} случайных комбинаций из {total_combinations} возможных")
print(f"Это всего {hp.n_iter/total_combinations*100:.4f}% от полного перебора!")
print("\nЛучшие параметры:", hp.best_params_)
print("Лучший f1 score:", hp.best_score_)

# Get top 5 results
top_5 = hp.get_top_results(5)
print("\nТоп-5 результатов:")
print(top_5)

# Compare with theoretical full grid search time
estimated_full_grid_time = hp.n_iter / total_combinations * 100 * 2  # Предположим 2 минуты на комбинацию
if estimated_full_grid_time > 60:
    hours = estimated_full_grid_time / 60
    time_str = f"{hours:.1f} часов"
else:
    time_str = f"{estimated_full_grid_time:.1f} минут"

print("\n" + "="*50)
print(f"ЭКОНОМИЯ ВРЕМЕНИ")
print("="*50)
print(f"Полный перебор всех комбинаций занял бы примерно {time_str}")
print(f"Случайный поиск выполнился за {hp.n_iter} комбинаций и нашел хорошие параметры!")
print("="*50)

# Clean up checkpoints after successful run
hp.clear_checkpoint()
print("\nЧекпоинт успешно удален.")

# Tip for users
print("\nСовет: Для очень больших пространств параметров начните со случайного поиска,")
print("затем используйте найденные лучшие параметры как основу для более детального поиска.")
