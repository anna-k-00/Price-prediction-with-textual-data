import pandas as pd
from IPython.display import HTML

class RealEstateAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self._validate_columns()
        self._process_data()
        
    def _validate_columns(self):
        """Проверяет наличие необходимых столбцов"""
        required_cols = {'price', 'id', 'predicted_price'}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            raise Warning(f"Отсутствуют необходимые столбцы: {missing}")
    
    def _process_data(self):
        """Обрабатывает данные и создает флаги"""
        # Рассчитываем относительную ошибку
        self.df['relative_error'] = abs(100 * (self.df['predicted_price'] - self.df['price']) / self.df['price'])
        
        # Создаем кликабельные ссылки
        self.df['link'] = self.df['id'].apply(
            lambda x: f'<a href="https://www.avito.ru/{x}" target="_blank">Открыть на Avito</a>'
        )
        
        # Флаг 2 (основные подозрительные случаи)
        self.df['flag_2'] = (
            ((self.df['price'] <= 50000000) & (self.df['relative_error'] >= 45)) | 
            ((self.df['price'] > 50000000) & (self.df['price'] <= 100000000) & (self.df['relative_error'] >= 75)))
        
        # Флаг 1 (очень подозрительные случаи)
        self.df['flag_1'] = (
            (self.df['price'] <= 100000000) & 
            (self.df['relative_error'] > 100))
    
    def fraud(self):
        """Возвращает очень подозрительные объявления (flag_1 == True)"""
        return self.df[self.df['flag_1']]
    
    def suspicious(self):
        """Возвращает подозрительные объявления (flag_2 == True)"""
        return self.df[self.df['flag_2']]
    
    def get_processed_data(self):
        """Возвращает все обработанные данные"""
        return self.df
