from tqdm import tqdm
import re
import pandas as pd
from functools import lru_cache
from symspellpy import SymSpell, Verbosity
from natasha import MorphVocab, NewsMorphTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import requests
from requests.exceptions import RequestException
from joblib import Parallel, delayed
import nltk
from tqdm import tqdm  # Use tqdm.auto for better compatibility
import pymorphy3 



# Ensure NLTK stopwords are downloaded
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, 
                 df, 
                 text_columns = None, 
                 fix_spelling = True, 
                 lemmatize_text = True, 
                 remove_stopwords = True, 
                 remove_punctuation = True,
                 remove_trash = True,
                 mask_nums=False,
                 n_jobs=-1):
        """
        Initialize the TextPreprocessor class.

        :param df: Input DataFrame containing text data.
        :param text_columns: List of column names to process. If None, all columns are processed.
        :param fix_spelling: If True, perform spell correction in the final step.
        :param lemmatize_text: If True, perform lemmatization after all other preprocessing steps.
        :param remove_stopwords: If True, remove stopwords after lemmatization.
        """
        self.mask_nums = mask_nums
        self.n_jobs = n_jobs
        self.df = df.copy()
        self.text_columns = text_columns if text_columns else df.columns
        self.fix_spelling = fix_spelling
        self.lemmatize_text = lemmatize_text
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_trash = remove_trash
        # Load stopwords in the main process
        self.stop_words = set(stopwords.words('russian')) if remove_stopwords else set()
        self.stop_words.difference_update({'нет', 'есть'})  # Optional: Keep certain stopwords
        self.stop_words.difference_update({'нет', 'есть'})
        # Initialize trash words patterns if remove_trash is True
        
        if self.remove_trash:
            self.unwanted_roots = {
                'звон', 'пиш', 'пис', 'посыл', 'слан', 'приезж', 'отправ', 'говор', 
                'зада', 'спраш', 'вопрос', 'сообщ', 'дава', 'отвеч', 'ответ', 
                'советова', 'смотр', 'узна', 'мож', 'мог', 'уточн', 'помог', 
                'сказ', 'помоч', 'говор'
            }
            
            self.politeness_phrases = {
                'здравствуйте', 'привет', 'добрый', 'день', 'утро', 'вечер', 'ночь',
                'спасибо', 'благодар', 'пожалуйста', 'прошу', 'будь', 'добр', 'любезн',
                'извините', 'простите', 'до свидания', 'пока', 'всего хорошего'
            }
            
            self.pronoun_tags = {'NPRO'}
            
            # Initialize morph analyzer
            self.morph = pymorphy3.MorphAnalyzer()
            
        # Initialize ordinal replacements
        self.ordinal_replacements = {
            # Именительный падеж (единственное число)
            r'\b1\s*-?\s*(?:ы[й]|й)\b': 'первый',
            r'\b2\s*-?\s*(?:ы[й]|й)\b': 'второй',
            r'\b3\s*-?\s*(?:ы[й]|й)\b': 'третий',
            r'\b4\s*-?\s*(?:ы[й]|й)\b': 'четвёртый',
            r'\b5\s*-?\s*(?:ы[й]|й)\b': 'пятый',
            r'\b6\s*-?\s*(?:ы[й]|й)\b': 'шестой',
            r'\b7\s*-?\s*(?:ы[й]|й)\b': 'седьмой',
            r'\b8\s*-?\s*(?:ы[й]|й)\b': 'восьмой',
            r'\b9\s*-?\s*(?:ы[й]|й)\b': 'девятый',
            r'\b10\s*-?\s*(?:ы[й]|й)\b': 'десятый',
            # Именительный падеж (множественное число)
            r'\b1\s*-?\s*(?:ы[е]|е)\b': 'первые',
            r'\b2\s*-?\s*(?:ы[е]|е)\b': 'вторые',
            r'\b3\s*-?\s*(?:ы[е]|е)\b': 'третьи',
            r'\b4\s*-?\s*(?:ы[е]|е)\b': 'четвёртые',
            r'\b5\s*-?\s*(?:ы[е]|е)\b': 'пятые',
            r'\b6\s*-?\s*(?:ы[е]|е)\b': 'шестые',
            r'\b7\s*-?\s*(?:ы[е]|е)\b': 'седьмые',
            r'\b8\s*-?\s*(?:ы[е]|е)\b': 'восьмые',
            r'\b9\s*-?\s*(?:ы[е]|е)\b': 'девятые',
            r'\b10\s*-?\s*(?:ы[е]|е)\b': 'десятые',

            # Родительный падеж (единственное число)
            r'\b1\s*-?\s*(?:ого|го)\b': 'первого',
            r'\b2\s*-?\s*(?:ого|го)\b': 'второго',
            r'\b3\s*-?\s*(?:ого|го)\b': 'третьего',
            r'\b4\s*-?\s*(?:ого|го)\b': 'четвёртого',
            r'\b5\s*-?\s*(?:ого|го)\b': 'пятого',
            r'\b6\s*-?\s*(?:ого|го)\b': 'шестого',
            r'\b7\s*-?\s*(?:ого|го)\b': 'седьмого',
            r'\b8\s*-?\s*(?:ого|го)\b': 'восьмого',
            r'\b9\s*-?\s*(?:ого|го)\b': 'девятого',
            r'\b10\s*-?\s*(?:ого|го)\b': 'десятого',

            # Родительный падеж (множественное число)
            r'\b1\s*-?\s*(?:ы[х]|х)\b': 'первых',
            r'\b2\s*-?\s*(?:ы[х]|х)\b': 'вторых',
            r'\b3\s*-?\s*(?:ы[х]|х)\b': 'третьих',
            r'\b4\s*-?\s*(?:ы[х]|х)\b': 'четвёртых',
            r'\b5\s*-?\s*(?:ы[х]|х)\b': 'пятых',
            r'\b6\s*-?\s*(?:ы[х]|х)\b': 'шестых',
            r'\b7\s*-?\s*(?:ы[х]|х)\b': 'седьмых',
            r'\b8\s*-?\s*(?:ы[х]|х)\b': 'восьмых',
            r'\b9\s*-?\s*(?:ы[х]|х)\b': 'девятых',
            r'\b10\s*-?\s*(?:ы[х]|х)\b': 'десятых',

            # Дательный падеж (единственное число)
            r'\b1\s*-?\s*(?:ому|му)\b': 'первому',
            r'\b2\s*-?\s*(?:ому|му)\b': 'второму',
            r'\b3\s*-?\s*(?:ому|му)\b': 'третьему',
            r'\b4\s*-?\s*(?:ому|му)\b': 'четвёртому',
            r'\b5\s*-?\s*(?:ому|му)\b': 'пятому',
            r'\b6\s*-?\s*(?:ому|му)\b': 'шестому',
            r'\b7\s*-?\s*(?:ому|му)\b': 'седьмому',
            r'\b8\s*-?\s*(?:ому|му)\b': 'восьмому',
            r'\b9\s*-?\s*(?:ому|му)\b': 'девятому',
            r'\b10\s*-?\s*(?:ому|му)\b': 'десятому',

            # Дательный падеж (множественное число)
            r'\b1\s*-?\s*(?:ы[м]|м)\b': 'первым',
            r'\b2\s*-?\s*(?:ы[м]|м)\b': 'вторым',
            r'\b3\s*-?\s*(?:ы[м]|м)\b': 'третьим',
            r'\b4\s*-?\s*(?:ы[м]|м)\b': 'четвёртым',
            r'\b5\s*-?\s*(?:ы[м]|м)\b': 'пятым',
            r'\b6\s*-?\s*(?:ы[м]|м)\b': 'шестым',
            r'\b7\s*-?\s*(?:ы[м]|м)\b': 'седьмым',
            r'\b8\s*-?\s*(?:ы[м]|м)\b': 'восьмым',
            r'\b9\s*-?\s*(?:ы[м]|м)\b': 'девятым',
            r'\b10\s*-?\s*(?:ы[м]|м)\b': 'десятым',

            # Винительный падеж (единственное число)
            r'\b1\s*-?\s*(?:ы[й]|й)\b': 'первый',
            r'\b2\s*-?\s*(?:ы[й]|й)\b': 'второй',
            r'\b3\s*-?\s*(?:ы[й]|й)\b': 'третий',
            r'\b4\s*-?\s*(?:ы[й]|й)\b': 'четвёртый',
            r'\b5\s*-?\s*(?:ы[й]|й)\b': 'пятый',
            r'\b6\s*-?\s*(?:ы[й]|й)\b': 'шестой',
            r'\b7\s*-?\s*(?:ы[й]|й)\b': 'седьмой',
            r'\b8\s*-?\s*(?:ы[й]|й)\b': 'восьмой',
            r'\b9\s*-?\s*(?:ы[й]|й)\b': 'девятый',
            r'\b10\s*-?\s*(?:ы[й]|й)\b': 'десятый',

            # Винительный падеж (множественное число)
            r'\b1\s*-?\s*(?:ы[е]|е)\b': 'первые',
            r'\b2\s*-?\s*(?:ы[е]|е)\b': 'вторые',
            r'\b3\s*-?\s*(?:ы[е]|е)\b': 'третьи',
            r'\b4\s*-?\s*(?:ы[е]|е)\b': 'четвёртые',
            r'\b5\s*-?\s*(?:ы[е]|е)\b': 'пятые',
            r'\b6\s*-?\s*(?:ы[е]|е)\b': 'шестые',
            r'\b7\s*-?\s*(?:ы[е]|е)\b': 'седьмые',
            r'\b8\s*-?\s*(?:ы[е]|е)\b': 'восьмые',
            r'\b9\s*-?\s*(?:ы[е]|е)\b': 'девятые',
            r'\b10\s*-?\s*(?:ы[е]|е)\b': 'десятые',

            # Творительный падеж (единственное число)
            r'\b1\s*-?\s*(?:ы[м]|м)\b': 'первым',
            r'\b2\s*-?\s*(?:ы[м]|м)\b': 'вторым',
            r'\b3\s*-?\s*(?:ы[м]|м)\b': 'третьим',
            r'\b4\s*-?\s*(?:ы[м]|м)\b': 'четвёртым',
            r'\b5\s*-?\s*(?:ы[м]|м)\b': 'пятым',
            r'\b6\s*-?\s*(?:ы[м]|м)\b': 'шестым',
            r'\b7\s*-?\s*(?:ы[м]|м)\b': 'седьмым',
            r'\b8\s*-?\s*(?:ы[м]|м)\b': 'восьмым',
            r'\b9\s*-?\s*(?:ы[м]|м)\b': 'девятым',
            r'\b10\s*-?\s*(?:ы[м]|м)\b': 'десятым',

            # Творительный падеж (множественное число)
            r'\b1\s*-?\s*(?:ы[ми]|ми)\b': 'первыми',
            r'\b2\s*-?\s*(?:ы[ми]|ми)\b': 'вторыми',
            r'\b3\s*-?\s*(?:ы[ми]|ми)\b': 'третьими',
            r'\b4\s*-?\s*(?:ы[ми]|ми)\b': 'четвёртыми',
            r'\b5\s*-?\s*(?:ы[ми]|ми)\b': 'пятыми',
            r'\b6\s*-?\s*(?:ы[ми]|ми)\b': 'шестыми',
            r'\b7\s*-?\s*(?:ы[ми]|ми)\b': 'седьмыми',
            r'\b8\s*-?\s*(?:ы[ми]|ми)\b': 'восьмыми',
            r'\b9\s*-?\s*(?:ы[ми]|ми)\b': 'девятыми',
            r'\b10\s*-?\s*(?:ы[ми]|ми)\b': 'десятыми',

            # Предложный падеж (единственное число)
            r'\b1\s*-?\s*(?:ом|м)\b': 'первом',
            r'\b2\s*-?\s*(?:ом|м)\b': 'втором',
            r'\b3\s*-?\s*(?:ом|м)\b': 'третьем',
            r'\b4\s*-?\s*(?:ом|м)\b': 'четвёртом',
            r'\b5\s*-?\s*(?:ом|м)\b': 'пятом',
            r'\b6\s*-?\s*(?:ом|м)\b': 'шестом',
            r'\b7\s*-?\s*(?:ом|м)\b': 'седьмом',
            r'\b8\s*-?\s*(?:ом|м)\b': 'восьмом',
            r'\b9\s*-?\s*(?:ом|м)\b': 'девятом',
            r'\b10\s*-?\s*(?:ом|м)\b': 'десятом',

            # Предложный падеж (множественное число)
            r'\b1\s*-?\s*(?:ы[х]|х)\b': 'первых',
            r'\b2\s*-?\s*(?:ы[х]|х)\b': 'вторых',
            r'\b3\s*-?\s*(?:ы[х]|х)\b': 'третьих',
            r'\b4\s*-?\s*(?:ы[х]|х)\b': 'четвёртых',
            r'\b5\s*-?\s*(?:ы[х]|х)\b': 'пятых',
            r'\b6\s*-?\s*(?:ы[х]|х)\b': 'шестых',
            r'\b7\s*-?\s*(?:ы[х]|х)\b': 'седьмых',
            r'\b8\s*-?\s*(?:ы[х]|х)\b': 'восьмых',
            r'\b9\s*-?\s*(?:ы[х]|х)\b': 'девятых',
            r'\b10\s*-?\s*(?:ы[х]|х)\b': 'десятых',

            # Additional expressions for "2"
            r'\bдвум\b': 'второму',
            r'\b2\s*-?\s*м\b': 'второму',
            r'\b2\s*-?\s*ум\b': 'второму',
            r'\bдвумя\b': 'вторым',
            r'\b2\s*-?\s*мя\b': 'вторым',
            r'\b2\s*-?\s*я\b': 'второй',
            r'\b2\s*-?\s*х\b': 'вторых',
            r'\b2\s*-?\s*ух\b': 'вторых',
        }        

    def remove_price_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        number_pattern = re.compile(r'\d+')
        
        price_keywords = ["цена", "цены", "ценой", "цене", "цену", "ценам", "цен", "ценами", "ценах"]
        
        rub_pattern = re.compile(r'\d+\s*р(уб)?\.?')
        
        million_pattern = re.compile(r'\d+\s*(млн\.?|миллиона?|миллионов?)')
        
        filtered_sentences = []
        for sentence in sentences:
            contains_number = number_pattern.search(sentence)
            
            if contains_number:
                contains_price = any(keyword in sentence for keyword in price_keywords)
                
                contains_rub = rub_pattern.search(sentence)
                
                contains_million = million_pattern.search(sentence)
                
                if contains_price or contains_rub or contains_million:
                    continue
            
            filtered_sentences.append(sentence)
        
        # Собираем отфильтрованные предложения в новый текст
        return ' '.join(filtered_sentences)
    
    def preprocess_text(self, text):
        # Step 1: Remove emojis and additional signs
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F700-\U0001F77F"  # Alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric shapes
            "\U0001F800-\U0001F8FF"  # Supplemental arrows
            "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
            "\U0001FA00-\U0001FA6F"  # Chess symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and pictographs
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "⌛⏏⏰⏹↘⏺→↔↓°↙⁉"  # Additional signs to remove
            "⌛⏏⏰⏹↘⏺→↔↓°↙⁉"  # Additional signs to remove
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)  # Remove emojis and additional signs

        # Step 2: Remove unwanted tags like \r, \\r, \n, \u, \\u
        text = re.sub(r'\\[rnut]|\u2028', ' ', text)

        # Step 3: Ensure sentences end with proper punctuation before \r or \n
        segments = re.split(r'([\r\n]+)', text)
        processed_segments = []
        for segment in segments:
            if re.match(r'[\r\n]+', segment):  # If the segment is \r or \n
                processed_segments.append(segment)
            else:
                # Ensure the segment ends with proper punctuation
                if not re.search(r'[.!,;?]$', segment):  # If no punctuation at the end
                    segment += '.'  # Add a period
                processed_segments.append(segment)
        text = ''.join(processed_segments)  # Join segments back together

        # Step 4: Remove extra spaces (not more than 1 space)
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 5: Add spaces around signs [](){}
        text = re.sub(r'(?<!\s)([\[\](){}])', r' \1', text)  # Add space before
        text = re.sub(r'([\[\]();{}])(?!\s)', r'\1 ', text)  # Add space after

        # Step 6: Replace underscores and pipes with spaces
        text = re.sub(r'[_|]', ' ', text)

        # Step 7: Replace ? ! by dot and remove repeated dots (e.g., ??? -> .)
        text = re.sub(r'[!?]+', '.', text)  # Replace ? and ! with a single dot
        text = re.sub(r'\.{2,}', '.', text)  # Remove repeated dots
        text = re.sub(r'\.{2,}', ',', text)  # Remove repeated comas

        # Step 8: Ensure there is always a space between a letter and a digit
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)  # Letter followed by digit
        text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)  # Digit followed by letter

        # Add space before hyphen if there is no space before it
        text = re.sub(r'(?<!\s)-', ' -', text)
        # Add space after hyphen if there is no space after it
        text = re.sub(r'-(?!\s)', '- ', text)
        
        # Step 10: Remove repeated spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space

        # Step 11: Replace repeated \ or / signs with a single instance
        text = re.sub(r'[\\/]{2,}', r'/', text)  # Replace repeated \ or / with a single /

        return text

    def preprocess_dots(self, text):
        # Step 1: Identify and unify coordinate formats
        def unify_coordinates(match):
            # Extract latitude and longitude
            lat, lon = match.group(1), match.group(2)
            
            # Handle decimal degrees with ',' as the decimal separator
            if ',' in lat or ',' in lon:
                lat = lat.replace(',', '.')
                lon = lon.replace(',', '.')
            
            # Handle degrees, minutes, seconds format
            if '°' in lat:
                # Convert DMS to decimal degrees
                def dms_to_decimal(dms):
                    parts = re.split('[°\'"]+', dms)
                    degrees = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    direction = parts[3]
                    decimal = degrees + minutes / 60 + seconds / 3600
                    if direction in ['S', 'W']:
                        decimal = -decimal
                    return decimal
                
                lat = dms_to_decimal(lat)
                lon = dms_to_decimal(lon)
            
            # Return unified format: decimal degrees with '.' as the separator
            return f"{lat}, {lon}"
        
        # Regex to match decimal degrees (e.g., 55.946885, 36.538426 or 55,749033, 38,328731)
        decimal_degree_pattern = re.compile(r'(\d+[.,]\d+),\s*(\d+[.,]\d+)')
        # Regex to match degrees, minutes, seconds (e.g., 55°33'3"N 37°17'46"E)
        dms_pattern = re.compile(r'(\d+°\d+\'\d+"[NS]),\s*(\d+°\d+\'\d+"[EW])')

        # Temporarily replace coordinates with placeholders
        coordinates = []
        def replace_with_placeholder(match):
            coordinates.append(match.group(0))
            return f'@@COORD_{len(coordinates) - 1}@@'
        
        # Replace all coordinate patterns with placeholders
        text = decimal_degree_pattern.sub(replace_with_placeholder, text)
        text = dms_pattern.sub(replace_with_placeholder, text)
        
        # Step 2: Replace '!' or '?' with '.'
        text = re.sub(r'[!?]', '.', text)
        
        # Step 3: Replace repeated '.' or ',' with a single symbol
        text = re.sub(r'([.,])\1+', r'\1', text)
        
        # Step 4: Handle numbers with thousand separators (e.g., 54.000.000)
        def replace_thousand_separators(match):
            # Remove all '.' or ',' used as thousand separators
            number = match.group(0).replace('.', '').replace(',', '')
            return number
        
        # Regex to match numbers with thousand separators (e.g., 54.000.000)
        text = re.sub(r'\b\d+([.,]\d{3})+\b', replace_thousand_separators, text)

        # Normalize large numbers with spaces (e.g., 1 000 000 → 1000000)
        text = re.sub(
            r'\b(\d{1,3}(?: \d{3})+)\b', 
            lambda m: m.group(1).replace(' ', ''), 
            text
        )
        
        # Step 5: Handle floating point numbers
        def replace_floating(match):
            # Replace ',' with '.' in floating point numbers
            number = match.group(0).replace(',', '.')
            return number
        
        # Regex to match floating point numbers (digits with '.' or ',')
        text = re.sub(r'\b(\d+[.,]\d+)(\w*)\b', replace_floating, text)
        
        # Step 6: Restore the preserved coordinates in unified format
        for i, coord in enumerate(coordinates):
            unified_coord = unify_coordinates(re.match(decimal_degree_pattern, coord) or re.match(dms_pattern, coord))
            text = text.replace(f'@@COORD_{i}@@', unified_coord)
        
        return text

    def replace_symbols_with_words(self, text):
        # Define the symbols and their corresponding replacements
        symbol_replacements = {
            r'≈': 'примерно',
            r'∼': 'примерно',
            r'±': 'примерно',
            r'~': 'примерно',
            r'<': 'менее',
            r'>': 'более',
            r'=': 'равно',
            r'%': 'процент',
            r'℅': 'процент',
            r'®': 'торговая марка',
            r'£': 'фунт',
            r'$': 'доллар',
            r'€': 'евро',
            r'₽': 'рубль',
            r'⌀': 'диаметр',
            r'√': 'номер',
            r'№': 'номер',
            r'°': '.',
            r'×': 'на' 
        }

        # Iterate over the symbols and perform replacements
        for symbol, replacement in symbol_replacements.items():
            # Add space before the symbol if it's not already there
            text = re.sub(r'(?<!\s)' + re.escape(symbol), f' {replacement}', text)
            # Add space after the symbol if it's not already there
            text = re.sub(re.escape(symbol) + r'(?!\s)', f'{replacement} ', text)
            # Remove the original symbol
            text = re.sub(re.escape(symbol), replacement, text)

        # Normalize spaces (remove extra spaces)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def replace_square_meter(self, text):
        # Updated pattern to include digits and optional whitespace before the unit
        pattern = r'(\d+\s*)(кв\s*м|м\s*кв|м\.кв|кв\.м|м\s*\.\s*кв|кв\s*\.\s*м|кв/м|м\s*/\s*кв|квм|мкв|м²|м2|м\^2|кв\.метр|кв\s*метр|кв\.метров|кв\s*метров|кв\.|кв)(?=\W)'
        return re.sub(pattern, r'\1м²', text)

    def replace_expressions(self, text):
        # Основные замены
        replacements = {
            r'(?<!\w)(личн(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) подсобн(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) хозяйств(?:о|а|у|ом|е|ы|ов|ам|ами|ах))(?!\w)': 'лпх',
        
                # СНТ (садовое некоммерческое товарищество)
                r'(?<!\w)(садов(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) некоммерческ(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) товариществ(?:о|а|у|ом|е|ы|ов|ам|ами|ах))(?!\w)': 'снт',
                
                # ДНП (дачное некоммерческое партнерство)
                r'(?<!\w)(дачн(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) некоммерческ(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) партнерств(?:о|а|у|ом|е|ы|ов|ам|ами|ах))(?!\w)': 'днп',
                
                # ИЖС (индивидуальное жилищное строительство)
                r'(?<!\w)(индивидуальн(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) жилищн(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) строительств(?:о|а|у|ом|е|ы|ов|ам|ами|ах))(?!\w)': 'ижс',
                
                # МКАД (московская кольцевая автомобильная дорога)
                r'(?<!\w)(московск(?:ая|ой|ую|ою|ие|их|им|ими) кольцев(?:ая|ой|ую|ою|ые|ых|ым|ыми) автомобильн(?:ая|ой|ую|ою|ые|ых|ым|ыми) дорог(?:а|и|у|ой|е|ам|ами|ах))(?!\w)': 'мкад',
                r'(?<!\w)(московск(?:ая|ой|ую|ою|ие|их|им|ими) кольцев(?:ая|ой|ую|ою|ые|ых|ым|ыми) авто дорог(?:а|и|у|ой|е|ам|ами|ах))(?!\w)': 'мкад',
                
                # ЦКАД (центральная кольцевая автомобильная дорога)
                r'(?<!\w)(центральн(?:ая|ой|ую|ою|ые|ых|ым|ыми) кольцев(?:ая|ой|ую|ою|ые|ых|ым|ыми) автомобильн(?:ая|ой|ую|ою|ые|ых|ым|ыми) дорог(?:а|и|у|ой|е|ам|ами|ах))(?!\w)': 'цкад',
                r'(?<!\w)(центральн(?:ая|ой|ую|ою|ые|ых|ым|ыми) кольцев(?:ая|ой|ую|ою|ые|ых|ым|ыми) авто дорог(?:а|и|у|ой|е|ам|ами|ах))(?!\w)': 'цкад',
                
                # ПМЖ (постоянное место жительства)
                r'(?<!\w)(постоянн(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) мест(?:о|а|у|ом|е|ы|ов|ам|ами|ах) жительств(?:а|у|ом|е|ы|ов|ам|ами|ах))(?!\w)': 'пмж',
                
                # Материнский капитал
                r'(?<!\w)(мат капитал|мат\. капитал|материнский капитал)(?!\w)': 'материнский_капитал',
                r'(?<!\w)(мат капиталом|мат\. капиталом|материнским капиталом)(?!\w)': 'материнский_капитал',


                # Коттеджный поселок
                r'(?<!\w)(кп|коттеджный\s*пос(елок)?|кот\s*пос(елок)?|коттедджный\s*пос(елок)?)(?!\w)': 'коттеджный_поселок',
                # Без отделки
                r'(без\s*отделки|нет\s*отделки|отделки\s*нет)': 'без_отделки',
                # С отделкой
                r'(с\s*отделкой)': 'с_отделкой',
                # Черновая отделка
                r'(черновая\s*отделка|отделка\s*черновая|черновой\s*отделки|черновой\s*отделке)': 'черновая_отделка',
                # Предчистовая отделка
                r'(предчистовая\s*отделка|пред\s*чистовая\s*отделка|предчистовой\s*отделке|пред\s*чистовой\s*отделке|предчистовой\s*отделки|пред\s*чистовой\s*отделки)': 'предчистовая_отделка',
                # Чистовая отделка
                r'(чистовая\s*отделка|чистовой\s*отделки|чистовой\s*отделке)': 'чистовая_отделка',
                # Теплый контур
                r'(теплый\s*контур|теплом\s*контуре|теплого\s*контура)': 'теплый_контур',
                # Холодный контур
                r'(холодный\s*контур|холодном\s*контуре|холодного\s*контура)': 'холодный_контур',
                
                # СНТ (садовое некоммерческое товарищество)
                r'(?<!\w)(снт|с\.н\.т|с н т)(?!\w)': 'снт',
                
                # СТ (садовое товарищество)
                r'(?<!\w)(с\.т|ст|с\\т|с/т)(?!\w)': 'ст',
                r'(?<!\w)(садов(?:ое|ого|ому|ом|ая|ой|ую|ою|ые|ых|ым|ыми) товариществ(?:о|а|у|ом|е|ы|ов|ам|ами|ах))(?!\w)': 'ст',
                
                # ДНП (дачное некоммерческое партнерство)
                r'(?<!\w)(днп|д\.н\.п|д н п)(?!\w)': 'днп',
                
                # ИЖС (индивидуальное жилищное строительство)
                r'(?<!\w)(ижс|и\.ж\.с|и ж с)(?!\w)': 'ижс',
                
                # Железнодорожный
                r'(?<!\w)(ж\\д|ж/д|жд|ж\.д|ж д)(?!\w)': 'железнодорожный',
                r'(?<!\w)(ж\\д|ж/д|жд|ж\.д|ж д)(?!\w)': 'железнодорожный',
                
                # Железобетонный
                r'(?<!\w)(ж\\б|ж/б|ж\.б|ж б)(?!\w)': 'железобетонный',
                
                # Так далее
                r'(?<!\w)(т\.д\.|тд|т\. д)(?!\w)': 'так далее',
                
                # Общая площадь
                r'(?<!\w)(о\.п\.|оп|о\. п|о\\п)(?!\w)': 'общая площадь',
                
                # Земельный участок
                r'(?<!\w)(з\.у\.|зу|з\. у|з\\у)(?!\w)': 'земельный участок',
                
                # Станция метро
                r'(?<!\w)(ст метро|ст\. метро)(?!\w)': 'станция метро',
                
                # Артикул
                r'(?<!\w)(арт|арт\.)(?!\w)': 'артикул',
                
                # Станция метро
                r'(?<!\w)(станция м\.)(?!\w)': 'станция метро',
                
                # Городской округ
                r'(?<!\w)(г\.о|г\. о|го)(?!\w)': 'городской округ',
                
                # Московская область
                r'(?<!\w)(мо|м\.о|м/о|м о)(?!\w)': 'московская область',
                
                # Агентство недвижимости
                r'(?<!\w)(ан|а\.н)(?!\w)': 'агентство недвижимости',
                
                # Улица
                r'(?<!\w)(ул)(?!\w)': 'улица',
                
                # Район
                r'(?<!\w)(р-н|р\.н|рн)(?!\w)': 'район',
                
                # Область
                r'(?<!\w)(обл)(?!\w)': 'область',
                
                # Шоссе
                r'(?<!\w)(ш)(?!\w)': 'шоссе',
                
                # Станция
                r'(?<!\w)(ст)(?!\w)': 'станция',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Условие 1: (до г пробел, до пробела не цифра) И (после г пробел и буква или точка и буква или пробел и точка и буква)
        text = re.sub(r'(\D\s)г(?=\s[а-яА-Я]|\.\s?[а-яА-Я])', r'\1город', text, flags=re.IGNORECASE)

        # Условие 2: (до г точка или запятая) И (после г пробел и буква или точка и буква или пробел и точка и буква)
        text = re.sub(r'([.,]\s?)г(?=\s[а-яА-Я]|\.\s?[а-яА-Я])', r'\1город', text, flags=re.IGNORECASE)

        # Условие 1: (до р пробел, до пробела не цифра) И (после р пробел и буква или точка и буква или пробел и точка и буква)
        text = re.sub(r'(\D\s)р(?=\s[а-яА-Я]|\.\s?[а-яА-Я])', r'\1река', text, flags=re.IGNORECASE)

        # Условие 2: (до р точка или запятая) И (после р пробел и буква или точка и буква или пробел и точка и буква)
        text = re.sub(r'([.,]\s?)р(?=\s[а-яА-Я]|\.\s?[а-яА-Я])', r'\1река', text, flags=re.IGNORECASE)
        text = re.sub(r'(\s|^)р\.(?=\s[а-яА-Я])', r'\1река', text, flags=re.IGNORECASE)

        # Условие 1: (до м пробел, до пробела не цифра) И (после м пробел и буква или точка и буква или пробел и точка и буква)
        text = re.sub(r'(\D\s)м(?=\s[а-яА-Я]|\.\s?[а-яА-Я])', r'\1метро', text, flags=re.IGNORECASE)
        
        text = re.sub(r'([.,]\s?)м(?=\s[а-яА-Я]|\.\s?[а-яА-Я])', r'\1метро', text, flags=re.IGNORECASE)
        
        # д → дом (только перед цифрами)
        text = re.sub(r'(?<=\s)д\.?(?=\s?\d)', 'дом', text, flags=re.IGNORECASE)

        text = re.sub(r'\bд\.?(?=\s|$)', 'деревня', text, flags=re.IGNORECASE)

        patterns = {
            # Именительный падеж (кто? что?)
            r'(?<!\w)(с(ан)?[ .\\/-]?узел)(?!\w)': 'санузел',
            r'(?<!\w)(с(ан)?[ .\\/-]?у[зд]?[ .\\/-]?)(?!\w)': 'санузел',  # для сокращений
            
            # Родительный падеж (кого? чего?)
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[а])(?!\w)': 'санузел',
            r'(?<!\w)(с(ан)?[ .\\/-]?у[зл]?[а]?[ .\\/-]?)(?!\w)': 'санузел',

            # Дательный падеж (кому? чему?)
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[у])(?!\w)': 'санузел',
            
            # Винительный падеж (кого? что?)
            r'(?<!\w)(с(ан)?[ .\\/-]?узел)(?!\w)': 'санузел',  # совпадает с именительным
            
            # Творительный падеж (кем? чем?)
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[о]?м)(?!\w)': 'санузел',
            r'(?<!\w)(с(ан)?[ .\\/-]?узл[о]?м)(?!\w)': 'санузел',
            
            # Предложный падеж (о ком? о чём?)
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[е])(?!\w)': 'санузел',
            
            # Множественное число
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[ы])(?!\w)': 'санузел',    # именительный
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[о]?в)(?!\w)': 'санузел',  # родительный
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[а]?м)(?!\w)': 'санузел',  # дательный
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[а]?ми)(?!\w)': 'санузел', # творительный
            r'(?<!\w)(с(ан)?[ .\\/-]?узел[а]?х)(?!\w)': 'санузел',  # предложный
        }
        for pattern, replacement in patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def replace_dim(self, text):
        # Pattern to match expressions like "number × number × number" with optional units
        pattern = re.compile(
            r'(\d+\.?\d*)\s*([*xх×/\\])\s*(\d+\.?\d*)\s*(метр[а-я]*|сантиметр[а-я]*|миллиметр[а-я]*)?\s*([*xх×/\\])?\s*(\d+\.?\d*)?\s*(метр[а-я]*|сантиметр[а-я]*|миллиметр[а-я]*)?'
        )

        # Function to replace matched expressions
        def replace_match(match):
            num1 = match.group(1)
            op1 = match.group(2)
            num2 = match.group(3)
            unit1 = match.group(4) or ''
            op2 = match.group(5)
            num3 = match.group(6)
            unit2 = match.group(7) or ''

            # If there is a third number, it's a triple product
            if num3:
                return f"{num1} на {num2} на {num3} {unit2}".strip()
            else:
                return f"{num1} на {num2} {unit1}".strip()

        # Replace all matched expressions in the string
        result = pattern.sub(replace_match, text)
        return result

    def replace_units(self, text):
        # Define the replacement patterns and their corresponding values
        replacements = {
            r'(?<=[/\\\d\s])(м3|м\^3|м\.куб|мкуб|м куб)\.?\b': 'кубический метр',
            r'(?<=[/\\\d\s])(мм|м\.м)\.?\b': 'миллиметр',
            r'(?<=[/\\\d\s])(см|с\.м)\.?\b': 'сантиметр',
            r'(?<=[/\\\d\s])(км|к\.м)\.?\b': 'километр',
            r'(?<=[/\\\d\s])(квт|клвт)\.?\b': 'киловатт',
            r'(?<=[/\\\d\s])(кв|кв)\.?\b': 'квадрат',
            r'(?<=[/\\\d\s])(ч|ч\.)\.?\b': 'час',
            r'(?<=[/\\\d\s])(г|г\.)\.?\b': 'год',
            r'(?<=[/\\\d\s])(мин)\.?\b': 'минута',
            r'(?<=[/\\\d\s])(сот)\.?\b': 'сотка',
            r'(?<=[/\\\d\s])(га)\.?\b': 'гектар',
            r'(?<=[/\\\d\s])(р|руб)\.?\b': 'рубль',
            r'(?<=[/\\\d\s])(тыс\.|т)\.?\b': 'тысяча',
            r'(?<=[/\\\d\s])(т\.р|тр)\.?\b': 'тысяча рублей',
            r'(?<=[/\\\d\s])(л)\.?\b': 'литр',
            r'(?<=[/\\\d\s])(э|эт)\.?\b': 'этаж',
            r'(?<=\d)(в\.|в,)\s*\.?\b': 'вольт',
            # Обновленное правило для метров с исключениями
            r'(?<!\S)(?<!__)(м)(?!\d|ро|\w)\.?\b': 'метр'
        }

        # Perform replacements in the order defined
        for pattern, replacement in replacements.items():
            # Use a lambda to ensure a space is added between the digit and the replacement
            text = re.sub(pattern, lambda m: f" {replacement}", text)

        return text

    def replace_ordinals(self, text):
        for pattern, replacement in self.ordinal_replacements.items():
            text = re.sub(pattern, replacement, text)
        return text

    def transform_text_mln_k(self, text):
        # Define patterns for thousands and millions
        thousand_patterns = re.compile(r'(\d+(?:\.\d+)?)\s?(тысяч[ауи]?|тыс\.?|тыс\.?\s?руб\.?|тр\.?\s?т\.?\s?р\.?\s?к\.?|к)\b')
        million_patterns = re.compile(r'(\d+(?:\.\d+)?)\s?(миллион[ауи]?|млн\.?|млн\.?\s?руб\.?|млн\.?\s?р\.?)\b')

        # Function to replace matched patterns for thousands
        def replace_thousand(match):
            number = float(match.group(1))
            return str(int(number * 1000) if number.is_integer() else number * 1000)

        # Function to replace matched patterns for millions
        def replace_million(match):
            number = float(match.group(1))
            return str(int(number * 1000000) if number.is_integer() else number * 1000000)

        # Replace thousand patterns
        text = thousand_patterns.sub(replace_thousand, text)

        # Replace million patterns
        text = million_patterns.sub(replace_million, text)

        return text

    def correct_text(self, text):
        url = "https://speller.yandex.net/services/spellservice.json/checkText"
        
        # Разделяем текст на части, если он слишком длинный
        max_length = 10000  # Максимальная длина текста для одного запроса
        if len(text) > max_length:
            parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            corrected_text = ""
            for part in parts:
                corrected_text += correct_text(part)  # Рекурсивно обрабатываем каждую часть
            return corrected_text
        
        # Используем POST-запрос для отправки текста
        data = {
            "text": text,
            "lang": "ru",
            "options": 518
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()  # Проверка на ошибки HTTP
        except RequestException as e:
            
            print(f"Ошибка при запросе к API: {e}")
            return text  # Возвращаем исходный текст в случае ошибки
        
        try:
            errors = response.json()
        except requests.JSONDecodeError:
            print("Ошибка декодирования JSON")
            return text  # Возвращаем исходный текст в случае ошибки декодирования
        
        corrected_text = text
        for error in reversed(errors):
            corrected_text = corrected_text[:error['pos']] + error['s'][0] + corrected_text[error['pos'] + error['len']:]
        return corrected_text
    
    def final_touch(self, text):
        def remove_brackets_and_quotes(text):
            # Remove all types of brackets
            text = re.sub(r'[\(\)\[\]\{\}\<\>]', '', text)
            # Remove all types of quotes
            text = re.sub(r'[\'\"\`\«\\»]', ' ', text)
            return text

        def add_spaces_between_digits_and_letters(text):
            # Add space between digit and letter (including Cyrillic letters)
            text = re.sub(r'(\d)([a-zA-Zа-яА-Я])', r'\1 \2', text)
            # Add space between letter (including Cyrillic letters) and digit
            text = re.sub(r'([a-zA-Zа-яА-Я])(\d)', r'\1 \2', text)
            return text

        def remove_repeated_punctuation(text):
            # Replace repeated punctuation marks with the first one
            text = re.sub(r'([.,!?;:-])\s*([.,!?;:-])+', r'\1', text)
            # Remove spaces between punctuation marks
            text = re.sub(r'([.,!?;:-])\s+([.,!?;:-])', r'\1', text)
            return text

        def add_space_after_punctuation(match):
            punctuation = match.group(2)  # Group 2: Punctuation (.,)
            before = match.group(3) or ''  # Group 3: Characters before punctuation (or empty string if None)
            after = match.group(4) or ''  # Group 4: Characters after punctuation (or empty string if None)

            # Check if the punctuation is between two digits (e.g., 51.8)
            if re.match(r'\d+\.\d+', before + punctuation + after):
                return match.group(0)  # No change

            # Check if the punctuation is part of a number followed by a word from ordinal_replacements (e.g., 15.седьмым)
            if re.match(r'\d+\.', before + punctuation) and any(after.startswith(word) for word in self.ordinal_replacements.values()):
                return match.group(0)  # No change

            # Otherwise, add a space after the punctuation
            return punctuation + ' '

        # Step 1: Remove brackets and quotes
        text = remove_brackets_and_quotes(text)

        # Step 2: Add spaces between digits and letters
        text = add_spaces_between_digits_and_letters(text)

        # Step 3: Remove repeated punctuation
        text = remove_repeated_punctuation(text)

        # Step 4: Remove emoji-like symbols
        text = re.sub(r'[\u2000-\u2FFF\U0001F000-\U0001F6FF]', '', text, flags=re.UNICODE)

        # Step 5: Replace `:` with a space
        text = re.sub(r':', ' ', text)

        # Step 6: Add spaces around `+` and replace it with 'плюс'
        text = re.sub(r'\s*\+\s*', ' плюс ', text)

        # Step 7: Replace `;` and `*` with a space
        text = re.sub(r'[;*//]', ' ', text)
        text = re.sub(r'-', '', text)

        # Step 8: Remove any space before `.` or `,`
        text = re.sub(r'\s+([.,])', r'\1', text)

        # Step 9: Add a space after `.` or `,` unless it is between two digits or part of a number followed by a word from ordinal_replacements
        text = re.sub(r'(\d+\.\d+)|([.,])(\S*)(\S*)', lambda m: m.group(0) if m.group(1) else add_space_after_punctuation(m), text)

        # Step 10: Normalize spaces (replace multiple spaces with a single space)
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 11: Spell correction (if enabled)
        if self.fix_spelling:
            text = self.correct_text(text)

        return text

    def should_remove_word(self, word):
        """Determine if a word should be removed based on trash patterns."""
        if not self.remove_trash:
            return False
            
        lower_word = word.lower()
        
        # Удаление мусорных слов вежливости
        if any(polite in lower_word for polite in self.politeness_phrases):
            return True
        
        # Проверка на местоимения и притяжательные местоимения
        parsed = self.morph.parse(word)[0]
        if any(tag in parsed.tag for tag in self.pronoun_tags) and not parsed.normal_form.isupper():
            return True
        
        # Проверка однокоренности
        normal_form = parsed.normal_form
        word_stem = re.sub(r'(ся|сь)$', '', normal_form)
        
        if any(word_stem.startswith(root) for root in self.unwanted_roots):
            return True
        
        return False

    def remove_trash_words(self, text):
        """Remove trash words from text if the option is enabled."""
        if not self.remove_trash:
            return text
            
        words = word_tokenize(text, language='russian')
        cleaned_words = [word for word in words if not self.should_remove_word(word)]
        return ' '.join(cleaned_words)

    
    def remove_russian_stopwords(self, text):
        """
        Remove Russian stopwords from the text.
        """
        # Tokenize the text
        tokens = word_tokenize(text, language='russian')
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        
        return ' '.join(filtered_tokens)

    def mask_numbers(self, text):
        # Заменяет все числа (в том числе заменённые в ordinal_replacements)
        # Важно: сначала заменить все слова-числительные из ordinal_replacements на тег [NUM]
        for word in set(self.ordinal_replacements.values()):
            text = re.sub(r'\b{}\b'.format(re.escape(word)), '[NUM]', text)
        # Теперь заменить любые цифры (целые, дробные)
        text = re.sub(r'\b\d+([.,]\d+)?\b', '[NUM]', text)
        return text
    

    def process_text(self, text):
        """
        Apply all preprocessing steps to a single text string.
        """
        # Initialize Mystem and SymSpell objects here to avoid serialization issues
        m = Mystem()
        spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        text = self.remove_price_sentences(text)
        text = self.preprocess_text(text)
        text = self.preprocess_dots(text)
        text = self.replace_symbols_with_words(text)
        text = self.replace_square_meter(text)
        text = self.replace_expressions(text)

        if self.mask_nums:
            text = self.mask_numbers(text)
            
        
        text = self.replace_dim(text)
        text = self.replace_units(text)
        

        text = self.replace_ordinals(text)
        text = self.transform_text_mln_k(text)

        
        text = self.final_touch(text)

        # Lemmatization (if enabled)
        if self.lemmatize_text:
            text = ''.join(m.lemmatize(text))

        if self.mask_nums:
            text = self.mask_numbers(text)

        # Remove stopwords (if enabled)
        if self.remove_stopwords:
            text = self.remove_russian_stopwords(text)
        
        # Remove trash words (if enabled)
        text = self.remove_trash_words(text)
        
        # Remove all punctuation (if enabled)
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        
        return text


    def process_dataframe(self):
        """
        Process all text columns in the DataFrame with parallel processing.
        """
        for col in self.text_columns:
            # Use joblib to parallelize the processing of rows
            self.df[col] = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.process_text)(text) for text in tqdm(self.df[col], desc=f"Processing {col}")
            )
        return self.df
