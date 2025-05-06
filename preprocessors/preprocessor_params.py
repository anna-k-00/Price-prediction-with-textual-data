import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import re

class DataProcessingPipeline:
    def __init__(self, df, norm_needed=True, log_needed=True, one_hot_only=True, 
                 train=True, outlier_bounds=None, scaler=None, lat_long_scaler=None):
        self.df = df
        self.norm_needed = norm_needed
        self.log_needed = log_needed
        self.one_hot_only = one_hot_only
        self.train = train
        self.outlier_bounds = outlier_bounds if outlier_bounds else {}
        self.fitted_outlier_bounds = None
        self.scaler = scaler  # Добавляем параметр для скалера
        self.lat_long_scaler = lat_long_scaler  # Добавляем параметр для скалера координат
        self.column_mapping = {
            'id': 'id',
            'Количество комнат': 'rooms',
            'Площадь дома': 'houseArea',
            'Площадь участка': 'landArea',
            'Этажей в доме': 'floors',
            'Категория земель': 'landCategory',
            'Год постройки': 'year',
            'Материал стен': 'wallMaterial',
            'Санузел': 'bathroom',
            'Ремонт': 'renovation',
            'Электричество': 'electricity',
            'Отопление': 'heating',
            'Водоснабжение': 'waterSupply',
            'Газ': 'gas',
            'Канализация': 'sewerage',
            'Интернет и ТВ': 'media',
            'Парковка': 'parking',
            'Транспортная доступность': 'transportAccessibility',
            'Способ продажи': 'saleMethod',
            'Расстояние от МКАД': 'distanceFromMkad',
            'address': 'address',
            'price': 'price',
            'description': 'description',
            'description_raw': 'description_raw',
            'seller': 'seller',
            'category': 'category',
            'lat': 'latitude',
            'lon': 'longitude',
            'Инфраструктура': 'infrastructure',
            'Расстояние до центра города': 'distanceToCityCenter',
            'Терраса или веранда': 'terrace',
            'Для отдыха': 'recreation',
            'Коммуникации': 'utilities'
        }
        self.mkad_coordinates = [
            (55.880829, 37.442726), (55.908630, 37.576799), (55.896677, 37.647310),
            (55.880013, 37.731544), (55.818116, 37.829557), (55.779645, 37.842390),
            (55.710884, 37.838127), (55.700, 37.850), (55.654, 37.850),
            (55.630, 37.830), (55.575329, 37.688232), (55.576377, 37.592020),
            (55.610493, 37.493170), (55.639903, 37.460022), (55.711899, 37.385268),
            (55.610270, 37.493332), (55.638694, 37.459579), (55.667098, 37.427542),
            (55.714460, 37.386541), (55.753475, 37.370334), (55.790803, 37.371478),
            (55.832914, 37.393789), (55.880757, 37.445849), (55.907951, 37.545137),
            (55.880829, 37.442726)
        ]
        self.mkad_coordinates_swapped = [(lon, lat) for lat, lon in self.mkad_coordinates]
        self.mkad_polygon = Polygon(self.mkad_coordinates_swapped)
        self.cities = [
            {"name": "Balashikha", "latitude": 55.8094, "longitude": 37.9581},
            {"name": "Khimki", "latitude": 55.8970, "longitude": 37.4297},
            {"name": "Podolsk", "latitude": 55.4242, "longitude": 37.5547},
            {"name": "Korolyov", "latitude": 55.9162, "longitude": 37.8265},
            {"name": "Mytishchi", "latitude": 55.9116, "longitude": 37.7307},
            {"name": "Lyubertsy", "latitude": 55.6784, "longitude": 37.8933},
            {"name": "Kolomna", "latitude": 55.0793, "longitude": 38.7783},
            {"name": "Elektrostal", "latitude": 55.7896, "longitude": 38.4467},
            {"name": "Odintsovo", "latitude": 55.6772, "longitude": 37.2775},
            {"name": "Zheleznodorozhny", "latitude": 55.7504, "longitude": 38.0166},
            {"name": "Serpukhov", "latitude": 54.9158, "longitude": 37.4111},
            {"name": "Shchyolkovo", "latitude": 55.9249, "longitude": 37.9722},
            {"name": "Dolgoprudny", "latitude": 55.9387, "longitude": 37.5019},
            {"name": "Domodedovo", "latitude": 55.4368, "longitude": 37.7661},
            {"name": "Ramenskoye", "latitude": 55.5669, "longitude": 38.2303},
            {"name": "Reutov", "latitude": 55.7586, "longitude": 37.8616},
            {"name": "Noginsk", "latitude": 55.8525, "longitude": 38.4388},
            {"name": "Pushkino", "latitude": 56.0105, "longitude": 37.8474},
            {"name": "Zhukovsky", "latitude": 55.6013, "longitude": 38.1115},
            {"name": "Krasnogorsk", "latitude": 55.8204, "longitude": 37.3302},
            {"name": "Voskresensk", "latitude": 55.3173, "longitude": 38.6526},
            {"name": "Sergiev Posad", "latitude": 56.3100, "longitude": 38.1326},
            {"name": "Orekhovo-Zuyevo", "latitude": 55.8067, "longitude": 38.9618},
            {"name": "Klin", "latitude": 56.3333, "longitude": 36.7333},
            {"name": "Chekhov", "latitude": 55.1527, "longitude": 37.4783},
            {"name": "Naro-Fominsk", "latitude": 55.3875, "longitude": 36.7333},
            {"name": "Lobnya", "latitude": 56.0127, "longitude": 37.4744},
            {"name": "Dubna", "latitude": 56.7333, "longitude": 37.1667},
            {"name": "Yegoryevsk", "latitude": 55.3843, "longitude": 39.0309},
            {"name": "Stupino", "latitude": 54.9008, "longitude": 38.0708},
            {"name": "Pavlovsky Posad", "latitude": 55.7819, "longitude": 38.6506},
            {"name": "Istra", "latitude": 55.9225, "longitude": 36.8647},
            {"name": "Fryazino", "latitude": 55.9606, "longitude": 38.0456},
            {"name": "Lytkarino", "latitude": 55.5833, "longitude": 37.9000},
            {"name": "Dzerzhinsky", "latitude": 55.6286, "longitude": 37.8547},
            {"name": "Kashira", "latitude": 54.8333, "longitude": 38.1500},
            {"name": "Protvino", "latitude": 54.8667, "longitude": 37.2167},
            {"name": "Troitsk", "latitude": 55.4833, "longitude": 37.3000},
            {"name": "Lukhovitsy", "latitude": 54.9667, "longitude": 39.0167},
            {"name": "Zaraysk", "latitude": 54.7667, "longitude": 38.8833},
            {"name": "Mozhaysk", "latitude": 55.5000, "longitude": 36.0333},
            {"name": "Volokolamsk", "latitude": 56.0333, "longitude": 35.9500},
            {"name": "Shatura", "latitude": 55.5667, "longitude": 39.5333},
            {"name": "Zvenigorod", "latitude": 55.7333, "longitude": 36.8500},
            {"name": "Roshal", "latitude": 55.6667, "longitude": 39.8833},
            {"name": "Kubinka", "latitude": 55.5667, "longitude": 36.7000},
            {"name": "Chernogolovka", "latitude": 56.0167, "longitude": 38.3833},
            {"name": "Krasnoarmeysk", "latitude": 56.1000, "longitude": 38.1333},
            {"name": "Elektrogorsk", "latitude": 55.8833, "longitude": 38.7833},
            {"name": "Vysokovsk", "latitude": 56.3167, "longitude": 36.5500},
            {"name": "Likino-Dulyovo", "latitude": 55.7167, "longitude": 38.9500},
        ]
        self.transformations = {
            'electricity': {'есть': 'yes'},
            'bathroom': {'в доме': 'inside', 'на улице': 'outside'},
            'waterSupply': {'скважина': 'borehole', 'колодец': 'well', 'центральное': 'central'},
            'sewerage': {'септик': 'septik', 'центральная': 'central', 'станция биоочистки': 'bio', 'выгребная яма': 'cesspool'},
            'saleMethod': {'возможна ипотека': 'mortgage', 'реализация на торгах': 'auction', 'продажа доли': 'part'},
            'gas': {'по границе участка': 'border', 'в доме': 'inhouse'},
            'transportAccessibility': {'остановка общественного транспорта': 'bus', 'асфальтированная дорога': 'road', 'железнодорожная станция': 'railway'},
            'infrastructure': {'детский сад': 'kindergarden', 'магазин': 'shop', 'аптека': 'pharmacy', 'школа': 'school'},
            'parking': {'гараж': 'garage', 'парковочное место': 'slot'},
            'terrace': {'есть': 'yes'},
            'media': {'Wi-Fi': 'wifi', 'телевидение': 'tv'},
            'recreation': {'бассейн': 'pool', 'баня или сауна': 'sauna'},
            'heating': {'газовое': 'gas', 'электрическое': 'electric', 'центральное': 'central', 'камин': 'fireplace', 'печь': 'stove', 'жидкотопливный котёл': 'liquidFuelBoiler', 'другое': 'other'},
            'category': {'Дачи': 'dacha', 'Дома': 'house', 'Коттеджи': 'cottage'},
            'seller': {'Риелтор': 'rieltor', 'Агентство': 'agency', 'Частное лицо': 'owner'},
            'renovation': {'косметический': 'cosmetic', 'евро': 'evro', 'дизайнерский': 'design', 'требует ремонта': 'requested'},
            'landCategory': {'садовое некоммерческое товарищество (СНТ)': 'snt', 'фермерское хозяйство': 'farm', 'Личное подсобное хозяйство (ЛПХ)': 'lph', 'дачное некоммерческое партнёрство (ДНП)': 'dnp', 'индивидуальное жилищное строительство (ИЖС)': 'izhs'},
            'wallMaterial': {'бревно': 'log', 'экспериментальные материалы': 'experimental', 'пеноблоки': 'foamBlock', 'металл': 'metal', 'сэндвич-панели': 'sandwichPanel', 'газоблоки': 'gasBlock', 'кирпич': 'brick', 'железобетонные панели': 'concretePanel', 'брус': 'timber'},
        }
        self.highways = {
                'алтуфьевское': 'altufyevskoye',
                'боровское': 'borovskoye',
                'быковское': 'bykovskoye',
                'варшавское': 'varshavskoye',
                'волоколамское': 'volokolamskoye',
                'горьковское': 'gorkovskoye',
                'дмитровское': 'dmitrovskoye',
                'егорьевское': 'yegoryevskoye',
                'ильинское': 'ilinskoye',
                'калужское': 'kaluzhskoye',
                'каширское': 'kashirskoye',
                'киевское': 'kiyevskoye',
                'куркинское': 'kurkinskoye',
                'ленинградское': 'leningradskoye',
                'минское': 'minskoye',
                'можайское': 'mozhayskoye',
                'новокаширское': 'novokashirskoye',
                'новорижское': 'novorizhskoye',
                'новорязанское': 'novoryazanskoye',
                'новосходненское': 'novoskhodnenskoye',
                'носовихинское': 'nosovikhinskoye',
                'осташковское': 'ostashkovskoye',
                'пятницкое': 'pyatnitskoye',
                'рогачёвское': 'rogachyovskoye',
                'рублёво-успенское': 'rublyovo-uspenskoye',
                'рублёвское': 'rublyovskoye',
                'рязанское': 'ryazanskoye',
                'симферопольское': 'simferopolskoye',
                'сколковское': 'skolkovskoye',
                'фряновское': 'fryanovskoye',
                'щёлковское': 'shchyolkovskoye',
                'ярославское': 'yaroslavskoye'
            }


    def preprocess_base(self):
        """Базовые преобразования без нормализации и удаления выбросов"""
        # Step 1: Rename columns
        self.df.rename(columns=self.column_mapping, inplace=True)

        # Step 2: Highway name restoring
        self.df['nearestHighway'] = self.df['address'].apply(self.find_highway)
        self.transformations['nearestHighway'] = {key: key for key in self.highways.values()}
        
        # Step 3: Drop rows
        self.df = self.drop_rows(self.df)

        # Step 4: Convert data types
        self.df = self.convert_data_types(self.df)

        # Step 5: Recalculate MKAD distance
        self.df = self.distance_from_mkad(self.df)

        # Step 6: Clean year of construction
        self.df = self.clean_year_of_construction(self.df)

        # Step 8: Calculate nearest city and distance
        self.df = self.calculate_nearest_city(self.df)
        self.transformations['nearestCity'] = {city['name']: city['name'] for city in self.cities}

        # Additional step - make columns lists for further functions
        self._update_object_columns(self.df)

        # Step 10: Categorical replacements and conversions
        self.df, self.label_encoders = self.categorical_replacements_and_convertations(self.df)

        return self.df

    def prepare_for_model(self):
        """Подготовка данных для модели: удаление выбросов + нормализация"""
        # Удаление выбросов
        if self.train:
            self.df = self.remove_outliers(self.df, columns=['price','houseArea', 'landArea'])
        else:
            self.df = self.remove_outliers(self.df, columns=['price','houseArea', 'landArea'])
        
        # Логарифмическое преобразование (если нужно)
        if self.log_needed:
            log_columns = ['houseArea', 'landArea', 'distanceFromMkad', 'distanceToCityKm', 'price']
            for col in log_columns:
                self.df[col] = np.log1p(self.df[col])
        
        # Нормализация (если нужно)
        if self.norm_needed:
            columns_to_normalize = ['houseArea', 'landArea', 'distanceFromMkad', 
                                  'year', 'distanceToCityKm', 'price']
            
            if self.train:
                # Для обучающих данных - создаем и обучаем скалеры
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.lat_long_scaler = MinMaxScaler(feature_range=(-1, 1))
                
                # Обучаем и применяем нормализацию для числовых признаков
                self.df[columns_to_normalize] = self.scaler.fit_transform(self.df[columns_to_normalize])
                
                # Обучаем и применяем нормализацию для координат
                self.df[['latitude', 'longitude']] = self.lat_long_scaler.fit_transform(
                    self.df[['latitude', 'longitude']]
                )
            else:
                # Для тестовых данных - проверяем наличие скалеров
                if self.scaler is None or self.lat_long_scaler is None:
                    raise ValueError("Для тестовых данных необходимо передать обученные скалеры")
                
                # Применяем нормализацию для числовых признаков
                self.df[columns_to_normalize] = self.scaler.transform(self.df[columns_to_normalize])
                
                # Применяем нормализацию для координат
                self.df[['latitude', 'longitude']] = self.lat_long_scaler.transform(
                    self.df[['latitude', 'longitude']]
                )
        
        # Возвращаем результаты
        if self.train:
            return {
                'processed_df': self.df,
                'outlier_bounds': self.fitted_outlier_bounds,
                'scaler': self.scaler,
                'lat_long_scaler': self.lat_long_scaler
            }
        else:
            return self.df

    def _apply_normalization(self, df):
        """Применяет нормализацию к данным"""
        if self.log_needed:
            df['houseArea'] = np.log1p(df['houseArea'])
            df['landArea'] = np.log1p(df['landArea'])
            df['distanceFromMkad'] = np.log1p(df['distanceFromMkad'] + 1)
            df['distanceToCityKm'] = np.log1p(df['distanceToCityKm'] + 1)
            df['price'] = np.log1p(df['price'])
    
        if self.norm_needed and self.scaler is not None and self.lat_long_scaler is not None:
            # Нормализация для числовых признаков
            columns_to_normalize = ['houseArea', 'landArea', 'distanceFromMkad', 
                                   'year', 'distanceToCityKm', 'price']
            df[columns_to_normalize] = self.scaler.transform(df[columns_to_normalize])
            
            # Нормализация для координат
            df[['latitude', 'longitude']] = self.lat_long_scaler.transform(
                df[['latitude', 'longitude']]
            )
        
        return df
        
    def process_for_ml(self):
        # description_data = self.df[['id', 'description']].set_index('id').copy()
        
        # Step 1: Rename columns
        self.df.rename(columns=self.column_mapping, inplace=True)

        # Step 2: Highway name restoring
        self.df['nearestHighway'] = self.df['address'].apply(self.find_highway)
        self.transformations['nearestHighway'] = {key: key for key in self.highways.values()}
        
        # Step 3: Drop rows
        self.df = self.drop_rows(self.df)

        # Step 4: Convert data types
        self.df = self.convert_data_types(self.df)

        # Step 5: Recalculate MKAD distance
        self.df = self.distance_from_mkad(self.df)

        # Step 6: Clean year of construction
        self.df = self.clean_year_of_construction(self.df)

        # Step 7: Remove outliers (теперь сохраняет или применяет границы)
        self.df = self.remove_outliers(self.df, columns=['price', 'houseArea', 'landArea'])

        # Step 8: Calculate nearest city and distance
        self.df = self.calculate_nearest_city(self.df)
        self.transformations['nearestCity'] = {city['name']: city['name'] for city in self.cities}
        
        # Step 9: Normalization and log transformation
        self.df = self.normalization_logtransformation(self.df, self.norm_needed, self.log_needed)

        # Additional step - make columns lists for further functions
        self._update_object_columns(self.df)

        # Step 10: Categorical replacements and conversions
        self.df, self.label_encoders = self.categorical_replacements_and_convertations(self.df)

        # self.df = self.df.join(description_data)

        return self.df

    def restore(self):
        # Step 1: Reverse categorical replacements and conversions
        df = self.reverse_categorical_replacements_and_convertations(self.df)

        # Step 2: Reverse normalization and log transformation
        df = self.reverse_normalization_logtransformation(self.df, self.norm_needed, self.log_needed)

        return df
    
    def find_highway(self, address):
        highway_pattern = re.compile(
            '|'.join(re.escape(highway) for highway in self.highways.keys()),
            flags=re.IGNORECASE
        )
        matches = highway_pattern.findall(address.lower())
        if matches:
            return self.highways[matches[0].lower()]
        return None

    def drop_rows(self, df):
        description_phrases = [
            'построим', 'потсрою', 'вашему проекту', 'вашему тз', 'аукцион', 'торги',
            'продам долю', '1/3 дома', '1/2 дома', '1/4 дома', '0.5 дома', 'половину дома',
            'треть дома', 'четверть дома', '1/3 дачи', '1/2 дачи', '1/4 дачи', '0.5 дачи',
            'половину дачи', 'треть дачи', 'четверть дачи'
        ]
        sale_methods_to_drop = ['продажа доли', 'реализация на торгах']

        for phrase in description_phrases:
            df = df[~df['description_raw'].str.contains(phrase, case=False, na=False)]

        for phrase in sale_methods_to_drop:
            df = df[~df['saleMethod'].str.contains(phrase, case=False, na=False)]

        df = df.drop(columns=['utilities', 'distanceToCityCenter', 'address']).set_index(['id'])
        df = df[df.landCategory != 'фермерское хозяйство']
        return df

    def convert_data_types(self, df):
        integer_columns = ['price']
        float_columns = ['houseArea', 'landArea', 'distanceFromMkad', 'latitude', 'longitude', 'year']

        for col in integer_columns:
            if not pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].str.replace(',', '.', regex=False).str.replace(' ', '', regex=False).astype(int)

        for col in float_columns:
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].str.replace(',', '.', regex=False).str.replace(' ', '', regex=False).astype(float)

        df['rooms'] = df['rooms'].replace({'Свободная планировка': '0', '10 и больше': '10'}).astype(int)
        df['floors'] = df['floors'].replace({'4 и больше': '4'}).astype(int)

        return df

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the Earth.
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        radius = 6371  # Earth's radius in kilometers
        distance = radius * c
        return distance

    def distance_to_polygon(self, latitude, longitude, polygon=None):
        """
        Calculate the distance from a point to a polygon (default is MKAD polygon).
        If the point is inside the polygon, the distance is 0.
        """
        if polygon is None:
            polygon = self.mkad_polygon

        point = Point(longitude, latitude)

        if polygon.contains(point):
            return 0

        nearest_point = nearest_points(point, polygon)[1]
        nearest_lat = nearest_point.y
        nearest_lon = nearest_point.x

        distance = self.haversine(latitude, longitude, nearest_lat, nearest_lon)
        return distance

    def distance_from_mkad(self, df):
        """
        Recalculate the distance from MKAD for each row in the DataFrame.
        """
        for index, row in df.iterrows():
            if not pd.isna(row['latitude']) and not pd.isna(row['longitude']):
                df.at[index, 'distanceFromMkad'] = self.distance_to_polygon(
                    row['latitude'], row['longitude']
                )
        return df

    def clean_year_of_construction(self, df):
        median_year = df['year'].median()
        df['yearIsNull'] = df['year'].isna()
        df['yearIsNull'] = df['yearIsNull'].astype(int)
        df['year'] = df['year'].fillna(median_year)
        
        # Replace null values with mode
        df['year'] = df['year'].fillna(median_year)
        
        # Replace 0 with mode
        df['year'] = np.where(
            df['year'] == 0,
            median_year,
            df['year']
        )
        
        # Replace values larger than 2025 with 2025
        df['year'] = df['year'].clip(upper=2025)
        
        # Function to handle year formats with regex
        def fix_year(year): 
            if pd.isna(year):
                return year
            year = int(year)
            if 25 < year < 100:  # e.g., 96 → 1996, 87 → 1987
                return 1900 + year
            elif 0 <= year <= 25:  # e.g., 2 → 2002, 12 → 2012, 23 → 2023
                return 2000 + year
            elif 100 <= year < 1000:  # e.g., 207 → 2007, 209 → 2009, 216 → 2016
                return 2000 + (year % 100)
            else:
                return year
        
        # Apply the fix_year function to the column
        df['year'] = df['year'].apply(fix_year)
        df['year']=df['year'].astype(int)
        return df
        # integer_columns.append('yearIsNull')

    
    def remove_outliers(self, df, columns, lower_percentile=0.01, upper_percentile=0.95):
        """
        Remove the lower and upper percentiles from multiple columns in a DataFrame.
        In train mode: calculates and saves bounds
        In apply mode: uses pre-calculated bounds
        """
        if self.train:
            # Режим обучения - вычисляем границы
            bounds = {}
            for column in columns:
                lower_bound = df[column].quantile(lower_percentile)
                upper_bound = df[column].quantile(upper_percentile)
                bounds[column] = (lower_bound, upper_bound)
            
            # Сохраняем вычисленные границы
            self.fitted_outlier_bounds = bounds
            
            # Фильтруем данные
            mask = pd.Series(True, index=df.index)
            for column, (lower_bound, upper_bound) in bounds.items():
                mask &= (df[column] >= lower_bound) & (df[column] <= upper_bound)
        else:
            # Режим применения - используем предварительно вычисленные границы
            if not self.outlier_bounds:
                raise ValueError("Outlier bounds must be provided in apply mode")
            
            mask = pd.Series(True, index=df.index)
            for column in columns:
                if column not in self.outlier_bounds:
                    raise ValueError(f"No bounds provided for column: {column}")
                
                lower_bound, upper_bound = self.outlier_bounds[column]
                mask &= (df[column] >= lower_bound) & (df[column] <= upper_bound)
        
        return df[mask]
        
    def nearest_city(self, latitude, longitude):
        min_distance = float("inf")
        nearestCity = None
        
        for city in self.cities:
            distance = self.haversine(latitude, longitude, city["latitude"], city["longitude"])
            if distance < min_distance:
                min_distance = distance
                nearestCity = city["name"]
        
        return min_distance, nearestCity

    def calculate_nearest_city(self, df):
        df[["distanceToCityKm", "nearestCity"]] = df.apply(
            lambda row: pd.Series(self.nearest_city(row["latitude"], row["longitude"])),
            axis=1
        )
        return df


    def normalization_logtransformation(self, df, norm_needed = True, log_needed = True):
        print(norm_needed, log_needed)
        """
        Apply log transformation and normalization to the DataFrame.
        - Latitude and longitude are normalized to a range of -1 to 1.
        - Other columns are normalized to a range of 0 to 1.
        """
        if log_needed:
            df['houseArea'] = np.log1p(df['houseArea'])
            df['landArea'] = np.log1p(df['landArea'])
            df['distanceFromMkad'] = np.log1p(df['distanceFromMkad'] + 1)
            df['distanceToCityKm'] = np.log1p(df['distanceToCityKm'] + 1)
            df['price'] = np.log1p(df['price'])

        if norm_needed:
            # Normalization for columns to [0, 1]
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            columns_to_normalize = ['houseArea', 'landArea', 'distanceFromMkad', 'year', 'distanceToCityKm', 'price']
            df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])

            # Normalization for latitude and longitude to [-1, 1]
            self.lat_long_scaler = MinMaxScaler(feature_range=(-1, 1))
            df[['latitude', 'longitude']] = self.lat_long_scaler.fit_transform(df[['latitude', 'longitude']])

        return df

        
    def _update_object_columns(self, df):
        # Update object_columns
        self.object_columns = [col for col in df.select_dtypes(include=['object']).columns 
                         if col != 'description' and col != 'description_raw']
    
        # Update columns_with_commas
        self.columns_with_commas = [col for col in self.object_columns if df[col].str.contains(',').any()]
    
        # Update columns_with_few_unique_values
        columns_with_few_unique_values = [col for col in self.object_columns if df[col].nunique() <= 2]
    
        # Update always_one_hot
        self.always_one_hot = self.columns_with_commas + columns_with_few_unique_values
    
        # Update not_always_one_hot
        self.not_always_one_hot = [col for col in self.object_columns if col not in self.always_one_hot]
    
        # Update columns_to_not_fill_null
        self.columns_to_not_fill_null = {
            'category': 'mode',
            'seller': 'mode',
            'renovation': 'mode',
            'landCategory': 'mode',
            'wallMaterial': 'mode'
        }

    
    def preprocess_rawa_cat(self, df):
        """
        Preprocess categorical columns by filling null values based on the specified method.
        """
        for column, method in self.columns_to_not_fill_null.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            
            if method == 'mode':
                fill_value = df[column].mode()[0]  # mode() returns a Series; take the first value
                df[column] = df[column].fillna(fill_value)
            else:
                raise ValueError(f"Unsupported fill method: '{method}'. Supported methods: 'mode'.")
        
        # Fill remaining object columns with 'нет'
        for column in [col for col in self.object_columns if col not in self.columns_to_not_fill_null.keys()]:
            df[column] = df[column].replace('нет', 'NaN')
            df[column] = df[column].fillna('NaN')
            
        return df
    
    def split_and_map_columns(self, df, column, mappings):
        """
        Split a column by commas and map its values to new binary columns.
        """
        # Split the column values by commas and create lists
        df[column] = df[column].str.split(',')
    
        # Create new columns for each unique value (excluding 'нет' and 'NaN')
        for index, row in df.iterrows():
            if isinstance(row[column], list):  # Check if the value is a list
                for value in row[column]:
                    value = value.strip()  # Remove any leading/trailing whitespace
                    
                    # Skip 'NaN' values
                    if value == 'NaN':
                        continue
                    
                    # Get the mapped value
                    mapped_value = mappings.get(column, {}).get(value, value)
                    new_column_name = f"{column}_{mapped_value}"
                    df.at[index, new_column_name] = 1  # Set the value to 1
    
        # Fill NaN values in the new columns with 0
        new_columns = [col for col in df.columns if col.startswith(f"{column}_")]
        df[new_columns] = df[new_columns].fillna(0).astype(int)
    
        # Drop the original column if no longer needed
        df.drop(columns=[column], inplace=True)
    
        return df
    
    def one_hot_encode_with_mappings(self, df, column, mappings, exclude_value='NaN'):
        """
        One-hot encode a column based on the provided mappings.
        """
        if column in mappings.keys():
            # Create new columns for each mapped value (excluding the excluded value)
            for value, mapped_value in mappings[column].items():
                if value == exclude_value:
                    continue  # Skip creating a column for the excluded value
                
                new_column_name = f"{column}_{mapped_value}"
                df[new_column_name] = (df[column] == value).astype(int)
            
            # Handle rows where the column value is the excluded value
            exclude_mask = df[column] == exclude_value
            if exclude_mask.any():
                for value, mapped_value in mappings[column].items():
                    if value == exclude_value:
                        continue  # Skip the excluded value
                    new_column_name = f"{column}_{mapped_value}"
                    df.loc[exclude_mask, new_column_name] = 0
            
            # Drop the original column
            df.drop(columns=[column], inplace=True)
        
        return df
    
    def categorical_replacements_and_convertations(self, df):
        """
        Handle categorical columns by either one-hot encoding or label encoding.
        """
        df = self.preprocess_rawa_cat(df)
        
        if self.one_hot_only is True:
            for col in self.columns_with_commas:
                df = self.split_and_map_columns(df, col, self.transformations)
            
            for col in [col for col in self.object_columns if col not in self.columns_with_commas]:
                df = self.one_hot_encode_with_mappings(df, col, self.transformations)
            return df, None
        else:
            label_encoders = {}
            for col in self.not_always_one_hot:
                df[col] = df[col].fillna('NaN')
                label_encoder = LabelEncoder()
                
                df[col] = label_encoder.fit_transform(df[col])
                
                if 'NaN' in label_encoder.classes_:
                    nan_index = list(label_encoder.classes_).index('NaN')
                    df[col] = df[col].replace(nan_index, 0)
                
                label_encoders[col] = label_encoder
            
            for col in [col for col in self.object_columns if col not in self.not_always_one_hot]:
                df = self.one_hot_encode_with_mappings(df, col, self.transformations)
            return df, label_encoders



    def reverse_categorical_replacements_and_convertations(self, df):
        """
        Reverse the categorical transformations applied to the DataFrame.
        """
        # Reverse label encoding
        if self.label_encoders:
            for col, label_encoder in self.label_encoders.items():
                if col in [col.split('_')[0] for col in df.columns if col in self.object_columns]:
                    # Inverse transform the encoded values
                    df[col] = label_encoder.inverse_transform(df[col])
                    
                    # If the column was transformed using mappings, apply the reverse mapping
                    if col in self.transformations.keys():
                        reverse_mapping = self.transformations[col]
                        df[col] = df[col].map(reverse_mapping).fillna(df[col])
    
        # Reverse one-hot encoding for columns not in columns_with_commas
        for col in df.columns:
            if '_' in col and col in self.object_columns:  # Check if the column is one-hot encoded
                original_col = col.split('_')[0]  # Extract the original column name
                
                # Skip columns that were split by commas
                if original_col in self.columns_with_commas:
                    continue
                
                # Initialize the original column if it doesn't exist
                if original_col not in df.columns:
                    df[original_col] = None
                
                # Combine one-hot encoded columns back into the original column
                for _, row in df.iterrows():
                    if row[col] == 1:
                        df.at[_, original_col] = col.split('_', 1)[1]  # Extract the value after the first underscore
                
                # Drop the one-hot encoded column
                df.drop(columns=[col], inplace=True)
    
        # Fill remaining object columns with 'NaN'
        df[[col for col in self.object_columns if col not in self.columns_with_commas and col in df.columns]] = df[
            [col for col in self.object_columns if col not in self.columns_with_commas and col in df.columns]
        ].fillna('NaN')
    
        return df
    
    def reverse_normalization_logtransformation(self, df, norm_needed = True, log_needed = True):
        """
        Reverse the log transformation and normalization applied to the DataFrame.
        - Latitude and longitude are denormalized from [-1, 1].
        - Other columns are denormalized from [0, 1].
        """
        print(norm_needed, log_needed)
        if norm_needed:
            # Reverse normalization for columns normalized to [0, 1]
            if self.scaler is None:
                raise ValueError("Scaler is not fitted. Call 'normalization_logtransformation' first.")
            
            columns_to_normalize = ['houseArea', 'landArea', 'distanceFromMkad', 'year', 'distanceToCityKm', 'price']
            df[columns_to_normalize] = self.scaler.inverse_transform(df[columns_to_normalize])

            # Reverse normalization for latitude and longitude normalized to [-1, 1]
            if self.lat_long_scaler is None:
                raise ValueError("Latitude/Longitude scaler is not fitted. Call 'normalization_logtransformation' first.")
            df[['latitude', 'longitude']] = self.lat_long_scaler.inverse_transform(df[['latitude', 'longitude']])
        if log_needed:
            # Reverse log transformation
            df['houseArea'] = np.expm1(df['houseArea'])
            df['landArea'] = np.expm1(df['landArea'])
            df['distanceFromMkad'] = np.expm1(df['distanceFromMkad']) - 1  # Subtract 1 to reverse the +1 adjustment
            df['distanceToCityKm'] = np.expm1(df['distanceToCityKm']) - 1  # Subtract 1 to reverse the +1 adjustment
            df['price'] = np.expm1(df['price'])

        return df
