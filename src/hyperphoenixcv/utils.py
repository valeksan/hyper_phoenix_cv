"""
Вспомогательные функции для HyperPhoenixCV.
"""

def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Проверяет, был ли оценщик обучен.
    
    Эта функция является упрощенной версией check_is_fitted из scikit-learn.
    
    Parameters:
    -----------
    estimator : estimator instance
        Оценщик для проверки.
        
    attributes : str, list or tuple of str, default=None
        Атрибуты, которые должны быть установлены после обучения.
        
    msg : str, default=None
        Сообщение об ошибке, которое будет выведено, если оценщик не обучен.
        
    all_or_any : callable, {all, any}, default=all
        Функция, которая будет использоваться для проверки атрибутов.
    """
    if attributes is not None:
        if isinstance(attributes, str):
            attributes = [attributes]
        if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
            if msg is None:
                msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this estimator."
                )
            raise ValueError(msg % {"name": type(estimator).__name__})
