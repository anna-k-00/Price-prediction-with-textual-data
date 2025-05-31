import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import requests
from io import BytesIO
from zipfile import ZipFile
import re
import h3
import json
from typing import Dict, Optional

class DataProcessingPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        norm_needed: bool = True,
        log_needed: bool = True,
        one_hot_only: bool = True,
        train: bool = True,
        outlier_bounds: Optional[Dict] = None,
        scaler: Optional[object] = None,
        lat_long_scaler: Optional[object] = None,
        use_hex_features: bool = True,
        hex_resolution: int = 9,
        hex_stats: Optional[pd.DataFrame] = None,
        size_group_stats: Optional[pd.DataFrame] = None,
        global_median_ppsm: Optional[float] = None,
        global_median_ppland: Optional[float] = None,
        global_median_price: Optional[float] = None,
        manual_text_params: bool = False,
        manual_text_params_path: Optional[str] = None,
    ):
        """
        Initialize the DataProcessingPipeline with the given DataFrame and processing parameters.
        
        Args:
            df: Input DataFrame containing the raw data.
            norm_needed: Whether to apply normalization.
            log_needed: Whether to apply log transformation.
            one_hot_only: Whether to use one-hot encoding exclusively.
            train: Whether the pipeline is in training mode.
            outlier_bounds: Precomputed bounds for outlier detection.
            scaler: Pre-trained scaler for normalization.
            lat_long_scaler: Pre-trained scaler for latitude/longitude.
            use_hex_features: Whether to use H3 hexagon features.
            hex_resolution: Resolution level for H3 hexagons.
            hex_stats: Precomputed hexagon statistics.
            size_group_stats: Precomputed size group statistics.
            global_median_ppsm: Global median price per square meter.
            global_median_ppland: Global median price per land area.
            global_median_price: Global median price.
            manual_text_params: Whether to use manual text features.
            manual_text_params_path: Path to JSON file with text feature rules.
        """
        self.df = df
        self.norm_needed = norm_needed
        self.log_needed = log_needed
        self.one_hot_only = one_hot_only
        self.train = train
        self.outlier_bounds = outlier_bounds if outlier_bounds else {}
        self.scaler = scaler
        self.lat_long_scaler = lat_long_scaler
        self.use_hex_features = use_hex_features
        self.hex_resolution = hex_resolution
        self.hex_stats = hex_stats
        self.size_group_stats = size_group_stats
        self.global_median_ppsm = global_median_ppsm
        self.global_median_ppland = global_median_ppland
        self.global_median_price = global_median_price
        self.cities_df = self._load_geonames_data()
        self._prepare_cities_kdtree()

        self.manual_text_params = manual_text_params
        self.manual_text_params_path = manual_text_params_path
        if self.manual_text_params:
            if not self.manual_text_params_path:
                raise ValueError("If manual_text_params=True, manual_text_params_path must be provided")
            with open(self.manual_text_params_path, 'r', encoding='utf-8') as f:
                self.manual_text_features_dict = json.load(f)
        else:
            self.manual_text_features_dict = {}

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
            'Коммуникации': 'utilities',
            'region': 'region',
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

    def _add_manual_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary text features based on rules from the provided JSON dictionary.
        
        Args:
            df: DataFrame containing the raw text data.
            
        Returns:
            DataFrame with added binary text features.
        """
        if not self.manual_text_params or 'description_raw' not in df.columns:
            return df
        df = df.copy()
        text = df['description_raw'].astype(str).str.lower()
        import re
        for key, values in self.manual_text_features_dict.items():
            pattern = '|'.join(re.escape(v.lower()) for v in values)
            df[f"text_{key}"] = text.str.contains(pattern, na=False).astype(int)
        return df
        
    def _load_geonames_data(self):
        """
        Load and preprocess city data from Geonames.
        
        Returns:
            DataFrame containing filtered city data.
        """
        url = "http://download.geonames.org/export/dump/RU.zip"
        response = requests.get(url)
        zipfile = ZipFile(BytesIO(response.content))

        with zipfile.open("RU.txt") as f:
            df = pd.read_csv(f, sep='\t', header=None, encoding='utf-8')

        columns = [
            'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
            'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code',
            'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation',
            'dem', 'timezone', 'modification_date'
        ]
        df.columns = columns

        cities_df = df[df['feature_code'].isin(['PPLC','PPLA','PPLA2'])]
        return cities_df[['name', 'latitude', 'longitude', 'feature_code', 'admin1_code']]

    def _prepare_cities_kdtree(self):
        """
        Prepare KDTree structures for efficient city lookup based on coordinates.
        """
        self.ppla_df = self.cities_df[self.cities_df['feature_code'].isin(['PPLC', 'PPLA'])]
        self.ppla2_df = self.cities_df[self.cities_df['feature_code'] == 'PPLA2']
        
        self.ppla_tree = KDTree(self.ppla_df[['latitude', 'longitude']].values)
        self.ppla2_tree = KDTree(self.ppla2_df[['latitude', 'longitude']].values)
        
        self.moscow_coords = (55.7558, 37.6176)
        self.spb_coords = (59.9343, 30.3351)

    def calculate_azimuth(self, point_lat, point_lon, center_lat, center_lon):
        """
        Calculate the azimuth angle from a center point to a target point.
        
        Args:
            point_lat: Latitude of the target point.
            point_lon: Longitude of the target point.
            center_lat: Latitude of the center point.
            center_lon: Longitude of the center point.
            
        Returns:
            Azimuth angle in degrees (0-360).
        """
        lon_diff = radians(point_lon - center_lon)
        lat1 = radians(center_lat)
        lat2 = radians(point_lat)
        
        x = sin(lon_diff) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon_diff)
        azimuth = degrees(atan2(x, y))
        
        return (azimuth + 360) % 360
    
    def get_reference_point_for_azimuth(self, region):
        """
        Determine the reference point for azimuth calculation based on region.
        
        Args:
            region: String containing the region name.
            
        Returns:
            Tuple of (latitude, longitude) or None if no specific reference point.
        """
        if 'московск' in region.lower():
            return self.moscow_coords
        elif 'ленинградск' in region.lower() or 'петербург' in region.lower():
            return self.spb_coords
        else:
            return None

    def calculate_nearest_cities(self, df):
        """
        Calculate distances to nearest cities and azimuth angles for all properties.
        
        Args:
            df: DataFrame containing property coordinates.
            
        Returns:
            DataFrame with added city distance and azimuth features.
        """
        try:
            if not hasattr(self, 'ppla_tree') or not hasattr(self, 'ppla2_tree'):
                raise ValueError("City search trees not initialized")
            
            coords = df[['latitude', 'longitude']].values
            
            dist_ppla, idx_ppla = self.ppla_tree.query(coords, k=1)
            ppla_info = self.ppla_df.iloc[idx_ppla[:, 0]]
            
            df['distanceToPPLA'] = [
                self.haversine(lat, lon, p_lat, p_lon) 
                for (lat, lon), (p_lat, p_lon) in zip(
                    coords, 
                    ppla_info[['latitude', 'longitude']].values
                )
            ]
            df['nearestPPLA'] = ppla_info['name'].values
            
            dist_ppla2, idx_ppla2 = self.ppla2_tree.query(coords, k=1)
            ppla2_info = self.ppla2_df.iloc[idx_ppla2[:, 0]]
            
            df['distanceToPPLA2'] = [
                self.haversine(lat, lon, p_lat, p_lon) 
                for (lat, lon), (p_lat, p_lon) in zip(
                    coords, 
                    ppla2_info[['latitude', 'longitude']].values
                )
            ]
            df['nearestPPLA2'] = ppla2_info['name'].values
            
            azimut_data = []
            for i in range(len(df)):
                row = df.iloc[i]
                region = row.get('region', '')
                ref_point = self.get_reference_point_for_azimuth(region)
                
                if ref_point is None:
                    nearest_idx = idx_ppla[i, 0]
                    ref_point = (
                        self.ppla_df.iloc[nearest_idx]['latitude'], 
                        self.ppla_df.iloc[nearest_idx]['longitude']
                    )
                
                azimuth = self.calculate_azimuth(row['latitude'], row['longitude'], *ref_point)
                azimut_data.append((sin(radians(azimuth)), cos(radians(azimuth))))
            
            df['azimut_sin'], df['azimut_cos'] = zip(*azimut_data)
            
            return df
        
        except Exception as e:
            raise ValueError(f"Error calculating nearest cities: {type(e).__name__}: {str(e)}")
    
    def preprocess_base(self):
        """
        Perform basic preprocessing steps without normalization or outlier removal.
        
        Returns:
            Preprocessed DataFrame.
        """
        if 'region' not in self.df.columns:
            self.df['region'] = 'Московская область'
            
        self.df.rename(columns=self.column_mapping, inplace=True)
        self.df = self.convert_data_types(self.df)
        
        if self.df[['latitude', 'longitude']].isnull().any().any():
            raise ValueError("Missing values found in coordinates")
        
        self.df = self.calculate_nearest_cities(self.df)
        self.df = self._add_manual_text_features(self.df)
        self.df = self.drop_rows(self.df)
        self.df = self.distance_from_mkad(self.df)
        self.df = self.clean_year_of_construction(self.df)
        self._update_object_columns(self.df)
        print(self.object_columns)
        self.df, self.label_encoders = self.categorical_replacements_and_convertations(self.df)
    
        return self.df

    def categorical_replacements_and_convertations(self, df):
        """
        Handle categorical columns by either one-hot encoding or label encoding.
        
        Args:
            df: DataFrame containing categorical columns.
            
        Returns:
            Tuple of (processed DataFrame, label encoders if used).
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

    
    def prepare_for_model(self):
        """
        Prepare data for modeling by marking outliers, adding spatial features, and applying transformations.
        
        Returns:
            Dictionary of processed data and trained objects (in train mode) or processed DataFrame (in apply mode).
        """
        self.df = self.df.reset_index()
        
        if self.train:
            self.df = self.mark_outliers(self.df, columns=['houseArea', 'landArea'])
        else:
            self.df = self.mark_outliers(self.df, columns=['houseArea', 'landArea'])
        
        if self.use_hex_features:
            self.df = self._calculate_hexagon_metrics(self.df)
        
        if self.log_needed:
            log_columns = ['houseArea', 'landArea', 'distanceFromMkad', 'distanceToPPLA', 'distanceToPPLA2', 'price']
            if self.use_hex_features:
                log_columns.extend([
                    'hex_price_median', 'hex_price_per_sqm', 'hex_price_per_land',
                    'neighbor_hex_price_median', 'neighbor_hex_price_per_sqm', 
                    'neighbor_hex_price_per_land'
                ])
            
            for col in log_columns:
                if col in self.df.columns:
                    self.df[col] = np.log1p(self.df[col])
        
        if self.norm_needed:
            if self.use_hex_features:
                columns_to_normalize = [
                    'houseArea', 'landArea', 'distanceFromMkad', 
                    'year', 'distanceToPPLA', 'distanceToPPLA2', 'price'
                ]
            else:
                columns_to_normalize = [
                    'houseArea', 'landArea', 'distanceFromMkad', 
                    'year', 'distanceToPPLA', 'distanceToPPLA2', 'hex_price_median', 'hex_price_per_sqm', 'hex_price_per_land',
                    'neighbor_hex_price_median', 'neighbor_hex_price_per_sqm',
                    'neighbor_hex_price_per_land','price'
                ]
            
            columns_to_normalize = [col for col in columns_to_normalize if col in self.df.columns]
            
            if self.train:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.lat_long_scaler = MinMaxScaler(feature_range=(-1, 1))
                
                self.df[columns_to_normalize] = self.scaler.fit_transform(self.df[columns_to_normalize])
                self.df[['latitude', 'longitude']] = self.lat_long_scaler.fit_transform(
                    self.df[['latitude', 'longitude']]
                )
            else:
                if self.scaler is None or self.lat_long_scaler is None:
                    raise ValueError("For test data, trained scalers must be provided")
                
                self.df[columns_to_normalize] = self.scaler.transform(self.df[columns_to_normalize])
                self.df[['latitude', 'longitude']] = self.lat_long_scaler.transform(
                    self.df[['latitude', 'longitude']]
                )
        
        if self.train:
            return {
                'processed_df': self.df.set_index('id'),
                'outlier_bounds': self.fitted_outlier_bounds,
                'scaler': self.scaler,
                'lat_long_scaler': self.lat_long_scaler,
                'hex_stats': self.hex_stats if self.use_hex_features else None,
                'global_median_ppsm': self.global_median_ppsm if self.use_hex_features else None,
                'global_median_ppland': self.global_median_ppland if self.use_hex_features else None,
                'global_median_price': self.global_median_price if self.use_hex_features else None
            }
        else:
            return self.df.set_index('id')

    def mark_outliers(self, df, columns, lower_percentile=0.01, upper_percentile=0.99):
        """
        Mark outliers in specified columns instead of removing them.
        
        Args:
            df: DataFrame to process.
            columns: List of columns to check for outliers.
            lower_percentile: Lower percentile threshold.
            upper_percentile: Upper percentile threshold.
            
        Returns:
            DataFrame with outlier flags added.
        """
        if self.train:
            bounds = {}
            for column in columns:
                lower_bound = df[column].quantile(lower_percentile)
                upper_bound = df[column].quantile(upper_percentile)
                bounds[column] = (lower_bound, upper_bound)
            
            self.fitted_outlier_bounds = bounds
            
            for column, (lower_bound, upper_bound) in bounds.items():
                df[f'is_{column}_outlier'] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
        else:
            if not self.outlier_bounds:
                raise ValueError("Outlier bounds must be provided in apply mode")
            
            for column in columns:
                if column not in self.outlier_bounds:
                    raise ValueError(f"No bounds provided for column: {column}")
                
                lower_bound, upper_bound = self.outlier_bounds[column]
                df[f'is_{column}_outlier'] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
        
        return df

    def _calculate_hexagon_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate H3 hexagon-based spatial features for the properties.
        
        Args:
            df: DataFrame containing property coordinates.
            
        Returns:
            DataFrame with added hexagon-based features.
        """
        if not self.use_hex_features:
            return df
    
        df['hex_id'] = df.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], self.hex_resolution),
            axis=1
        )
    
        if self.train:
            non_outliers_mask = ~(
                                df['is_houseArea_outlier'].astype(bool) | 
                                df['is_landArea_outlier'].astype(bool)
            )
            df_non_outliers = df[non_outliers_mask]
        else:
            df_non_outliers = df
    
        hex_counts = df_non_outliers['hex_id'].value_counts().rename('hex_property_count')
        
        hex_stats = df_non_outliers.groupby('hex_id').agg({
            'price': 'median',
            'houseArea': 'median',
            'landArea': 'median'
        }).rename(columns={
            'price': 'hex_price_median',
            'houseArea': 'hex_median_house_area',
            'landArea': 'hex_median_land_area'
        })
    
        hex_stats = hex_stats.join(hex_counts)
    
        hex_stats['hex_price_per_sqm'] = (
            hex_stats['hex_price_median'] / 
            hex_stats['hex_median_house_area'].replace(0, np.nan)
        )
        hex_stats['hex_price_per_land'] = (
            hex_stats['hex_price_median'] / 
            hex_stats['hex_median_land_area'].replace(0, np.nan)
        )
    
        if self.train:
            df_non_outliers['price_per_sqm'] = np.where(
                df_non_outliers['houseArea'] > 0,
                df_non_outliers['price'] / df_non_outliers['houseArea'],
                np.nan
            )
            df_non_outliers['price_per_land'] = np.where(
                df_non_outliers['landArea'] > 0,
                df_non_outliers['price'] / df_non_outliers['landArea'],
                np.nan
            )
            self.global_median_ppsm = df_non_outliers['price_per_sqm'].median()
            self.global_median_ppland = df_non_outliers['price_per_land'].median()
            self.global_median_price = df_non_outliers['price'].median()
    
        hex_stats = hex_stats.fillna({
            'hex_price_median': self.global_median_price,
            'hex_price_per_sqm': self.global_median_ppsm,
            'hex_price_per_land': self.global_median_ppland,
            'hex_median_house_area': df_non_outliers['houseArea'].median(),
            'hex_median_land_area': df_non_outliers['landArea'].median(),
            'hex_property_count': 0
        })
    
        self.hex_stats = hex_stats.copy()
    
        hex_stats_reset = hex_stats.reset_index()
        
        df = df.merge(
            hex_stats_reset,
            on='hex_id',
            how='left',
            suffixes=('', '_y')
        )
    
        outlier_mask = (
            df['is_houseArea_outlier'].astype(bool) |
            df['is_landArea_outlier'].astype(bool)
        )
    
        hex_metric_cols = [
            'hex_price_median', 'hex_price_per_sqm', 'hex_price_per_land',
            'hex_median_house_area', 'hex_median_land_area', 'hex_property_count'
        ]
        
        for col in hex_metric_cols:
            df.loc[outlier_mask & df[col].isna(), col] = (
                self.global_median_price if 'price_median' in col else
                self.global_median_ppsm if 'sqm' in col else
                self.global_median_ppland if 'land' in col else
                df_non_outliers['houseArea'].median() if 'house_area' in col else
                df_non_outliers['landArea'].median() if 'land_area' in col else
                0
            )
    
        neighbor_stats = []
        for hex_id in hex_stats.index:
            neighbors = h3.grid_disk(hex_id, 1)
            valid_neighbors = [n for n in neighbors if n != hex_id and n in hex_stats.index]
            
            if valid_neighbors:
                neighbor_data = hex_stats.loc[valid_neighbors]
                stats = {
                    'hex_id': hex_id,
                    'neighbor_hex_price_median': neighbor_data['hex_price_median'].median(),
                    'neighbor_hex_price_per_sqm': (
                        neighbor_data['hex_price_median'].sum() / 
                        neighbor_data['hex_median_house_area'].sum()
                    ),
                    'neighbor_hex_price_per_land': (
                        neighbor_data['hex_price_median'].sum() / 
                        neighbor_data['hex_median_land_area'].sum()
                    )
                }
            else:
                stats = {
                    'hex_id': hex_id,
                    'neighbor_hex_price_median': self.global_median_price,
                    'neighbor_hex_price_per_sqm': self.global_median_ppsm,
                    'neighbor_hex_price_per_land': self.global_median_ppland
                }
            neighbor_stats.append(stats)
    
        neighbor_stats_df = pd.DataFrame(neighbor_stats)
        df = df.merge(neighbor_stats_df, on='hex_id', how='left')
    
        for col in ['neighbor_hex_price_median', 'neighbor_hex_price_per_sqm', 'neighbor_hex_price_per_land']:
            df.loc[outlier_mask & df[col].isna(), col] = (
                self.global_median_price if 'median' in col else
                self.global_median_ppsm if 'sqm' in col else
                self.global_median_ppland
            )
    
        drop_cols = ['hex_id']
        for col in df.columns:
            if col.endswith('_y'):
                original_col = col[:-2]
                if original_col in df.columns:
                    df[original_col] = df[original_col].fillna(df[col])
                    drop_cols.append(col)
        
        return df.drop(columns=drop_cols, errors='ignore')
        
    def process_for_ml(self):
        self.df.rename(columns=self.column_mapping, inplace=True)
        self.df['nearestHighway'] = self.df['address'].apply(self.find_highway)
        self.transformations['nearestHighway'] = {key: key for key in self.highways.values()}
        self.df = self.drop_rows(self.df)
        self.df = self.convert_data_types(self.df)
        self.df = self.distance_from_mkad(self.df)
        self.df = self.clean_year_of_construction(self.df)
        self.df = self.remove_outliers(self.df, columns=['price', 'houseArea', 'landArea'])
        self.df = self.calculate_nearest_city(self.df)
        self.transformations['nearestCity'] = {city['name']: city['name'] for city in self.cities}
        self.df = self.normalization_logtransformation(self.df, self.norm_needed, self.log_needed)
        self._update_object_columns(self.df)
        self.df, self.label_encoders = self.categorical_replacements_and_convertations(self.df)

        return self.df

    def restore(self):
        df = self.reverse_categorical_replacements_and_convertations(self.df)
        df = self.reverse_normalization_logtransformation(self.df, self.norm_needed, self.log_needed)

        return df
    
    def find_highway(self, address):
        if pd.isna(address):
            return None
            
        highway_pattern = re.compile(
            '|'.join(re.escape(highway) for highway in self.highways.keys()),
            flags=re.IGNORECASE
        )
        matches = highway_pattern.findall(str(address).lower())
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
        """
        Convert columns to appropriate data types with coordinate validation.
        
        Args:
            df: DataFrame to process.
            
        Returns:
            DataFrame with converted data types.
        """
        required_columns = ['latitude', 'longitude', 'price', 'houseArea', 'landArea', 'year']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
        for col in ['latitude', 'longitude']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                raise ValueError(f"Invalid values in column {col}")
    
        integer_columns = ['price']
        float_columns = ['houseArea', 'landArea', 'distanceFromMkad', 'year']
    
        for col in integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    
        if 'rooms' in df.columns:
            df['rooms'] = df['rooms'].replace({'Свободная планировка': '0', '10 и больше': '10'}).astype(int)
        
        if 'floors' in df.columns:
            df['floors'] = df['floors'].replace({'4 и больше': '4'}).astype(int)
    
        return df

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the Earth.
        
        Args:
            lat1: Latitude of point 1.
            lon1: Longitude of point 1.
            lat2: Latitude of point 2.
            lon2: Longitude of point 2.
            
        Returns:
            Distance in kilometers.
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        radius = 6371
        distance = radius * c
        return distance

    def distance_to_polygon(self, latitude, longitude, polygon=None):
        """
        Calculate distance from a point to a polygon (default is MKAD polygon).
        
        Args:
            latitude: Point latitude.
            longitude: Point longitude.
            polygon: Optional custom polygon (defaults to MKAD polygon).
            
        Returns:
            Distance in kilometers (0 if point is inside polygon).
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
        Recalculate distance from MKAD for each property in the DataFrame.
        
        Args:
            df: DataFrame containing property coordinates.
            
        Returns:
            DataFrame with updated distanceFromMkad values.
        """
        for index, row in df.iterrows():
            if not pd.isna(row['latitude']) and not pd.isna(row['longitude']):
                df.at[index, 'distanceFromMkad'] = self.distance_to_polygon(
                    row['latitude'], row['longitude']
                )
        return df

    def clean_year_of_construction(self, df):
        """
        Clean and standardize year of construction values.
        
        Args:
            df: DataFrame containing year values.
            
        Returns:
            DataFrame with cleaned year values.
        """
        median_year = df['year'].median()
        df['yearIsNull'] = df['year'].isna()
        df['yearIsNull'] = df['yearIsNull'].astype(int)
        df['year'] = df['year'].fillna(median_year)
        df['year'] = df['year'].fillna(median_year)
        
        df['year'] = np.where(
            df['year'] == 0,
            median_year,
            df['year']
        )
        
        df['year'] = df['year'].clip(upper=2025)
        
        def fix_year(year): 
            if pd.isna(year):
                return year
            year = int(year)
            if 25 < year < 100:
                return 1900 + year
            elif 0 <= year <= 25:
                return 2000 + year
            elif 100 <= year < 1000:
                return 2000 + (year % 100)
            else:
                return year
        
        df['year'] = df['year'].apply(fix_year)
        df['year']=df['year'].astype(int)
        return df
    
    def remove_outliers(self, df, columns, lower_percentile=0.01, upper_percentile=0.99):
        """
        Remove outliers from specified columns based on percentile thresholds.
        
        Args:
            df: DataFrame to process.
            columns: List of columns to check for outliers.
            lower_percentile: Lower percentile threshold.
            upper_percentile: Upper percentile threshold.
            
        Returns:
            Filtered DataFrame with outliers removed.
        """
        if self.train:
            bounds = {}
            for column in columns:
                lower_bound = df[column].quantile(lower_percentile)
                upper_bound = df[column].quantile(upper_percentile)
                bounds[column] = (lower_bound, upper_bound)
            
            self.fitted_outlier_bounds = bounds
            
            mask = pd.Series(True, index=df.index)
            for column, (lower_bound, upper_bound) in bounds.items():
                mask &= (df[column] >= lower_bound) & (df[column] <= upper_bound)
        else:
            if not self.outlier_bounds:
                raise ValueError("Outlier bounds must be provided in apply mode")
            
            mask = pd.Series(True, index=df.index)
            for column in columns:
                if column not in self.outlier_bounds:
                    raise ValueError(f"No bounds provided for column: {column}")
                
                lower_bound, upper_bound = self.outlier_bounds[column]
                mask &= (df[column] >= lower_bound) & (df[column] <= upper_bound)
        
        return df[mask]


    def normalization_logtransformation(self, df, norm_needed = True, log_needed = True):
        """
        Apply log transformation and normalization to the DataFrame.
        
        Args:
            df: DataFrame to transform.
            norm_needed: Whether to apply normalization.
            log_needed: Whether to apply log transformation.
            
        Returns:
            Transformed DataFrame.
        """
        print(norm_needed, log_needed)
        if log_needed:
            df['houseArea'] = np.log1p(df['houseArea'])
            df['landArea'] = np.log1p(df['landArea'])
            df['distanceFromMkad'] = np.log1p(df['distanceFromMkad'] + 1)
            df['distanceToCityKm'] = np.log1p(df['distanceToCityKm'] + 1)
            df['price'] = np.log1p(df['price'])

        if norm_needed:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            columns_to_normalize = ['houseArea', 'landArea', 'distanceFromMkad', 'year', 'distanceToCityKm', 'price']
            df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])

            self.lat_long_scaler = MinMaxScaler(feature_range=(-1, 1))
            df[['latitude', 'longitude']] = self.lat_long_scaler.fit_transform(df[['latitude', 'longitude']])

        return df

        
    def _update_object_columns(self, df):
        """
        Update lists of object columns and their processing categories.
        """
        self.object_columns = [
            col for col in df.select_dtypes(include=['object', 'category']).columns 
            if col not in ['description', 'description_raw', 'azimut_sin', 'azimut_cos']
        ]
            
        self.columns_with_commas = [col for col in self.object_columns if df[col].str.contains(',').any()]
    
        columns_with_few_unique_values = [col for col in self.object_columns if df[col].nunique() <= 2]
    
        self.always_one_hot = self.columns_with_commas + columns_with_few_unique_values
    
        self.not_always_one_hot = [col for col in self.object_columns if col not in self.always_one_hot]
    
        self.columns_to_not_fill_null = {
            'category': 'mode',
            'seller': 'mode',
            'renovation': 'mode',
            'landCategory': 'mode',
            'wallMaterial': 'mode'
        }

    
    def preprocess_rawa_cat(self, df):
        """
        Preprocess categorical columns by filling null values appropriately.
        
        Args:
            df: DataFrame containing categorical columns.
            
        Returns:
            DataFrame with filled null values.
        """
        for column, method in self.columns_to_not_fill_null.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            
            if method == 'mode':
                fill_value = df[column].mode()[0]
                df[column] = df[column].fillna(fill_value)
            else:
                raise ValueError(f"Unsupported fill method: '{method}'. Supported methods: 'mode'.")
        
        for column in [col for col in self.object_columns if col not in self.columns_to_not_fill_null.keys()]:
            df[column] = df[column].replace('нет', 'NaN')
            df[column] = df[column].fillna('NaN')
            
        return df
    
    def split_and_map_columns(self, df, column, mappings):
        """
        Split comma-separated values and create binary columns for each value.
        
        Args:
            df: DataFrame containing the column to split.
            column: Name of the column to process.
            mappings: Dictionary of value mappings.
            
        Returns:
            DataFrame with original column replaced by binary columns.
        """
        df[column] = df[column].str.split(',')
    
        for index, row in df.iterrows():
            if isinstance(row[column], list):
                for value in row[column]:
                    value = value.strip()
                    
                    if value == 'NaN':
                        continue
                    
                    mapped_value = mappings.get(column, {}).get(value, value)
                    new_column_name = f"{column}_{mapped_value}"
                    df.at[index, new_column_name] = 1
    
        new_columns = [col for col in df.columns if col.startswith(f"{column}_")]
        df[new_columns] = df[new_columns].fillna(0).astype(int)
    
        df.drop(columns=[column], inplace=True)
    
        return df
    
    def one_hot_encode_with_mappings(self, df, column, mappings, exclude_value='NaN'):
        """
        One-hot encode a column using provided value mappings.
        
        Args:
            df: DataFrame containing the column to encode.
            column: Name of the column to process.
            mappings: Dictionary of value mappings.
            exclude_value: Value to exclude from encoding.
            
        Returns:
            DataFrame with original column replaced by binary columns.
        """
        if column in mappings.keys():
            for value, mapped_value in mappings[column].items():
                if value == exclude_value:
                    continue
                
                new_column_name = f"{column}_{mapped_value}"
                df[new_column_name] = (df[column] == value).astype(int)
            
            exclude_mask = df[column] == exclude_value
            if exclude_mask.any():
                for value, mapped_value in mappings[column].items():
                    if value == exclude_value:
                        continue
                    new_column_name = f"{column}_{mapped_value}"
                    df.loc[exclude_mask, new_column_name] = 0
            
            df.drop(columns=[column], inplace=True)
        else:
            if pd.api.types.is_categorical_dtype(df[column]):
                categories = df[column].cat.categories
            else:
                categories = df[column].unique()
            
            for category in categories:
                if category == exclude_value:
                    continue
                new_column_name = f"{column}_{category}"
                df[new_column_name] = (df[column] == category).astype(int)
            
            df.drop(columns=[column], inplace=True)
        
        return df
    
    def preprocess_rawa_cat(self, df):
        """
        Preprocess categorical columns by filling null values appropriately.
        
        Args:
            df: DataFrame containing categorical columns.
            
        Returns:
            DataFrame with filled null values.
        """
        for column, method in self.columns_to_not_fill_null.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            
            if method == 'mode':
                fill_value = df[column].mode()[0]
                df[column] = df[column].fillna(fill_value)
            else:
                raise ValueError(f"Unsupported fill method: '{method}'. Supported methods: 'mode'.")
        
        for column in [col for col in self.object_columns 
                      if col not in self.columns_to_not_fill_null.keys() 
                      and not pd.api.types.is_categorical_dtype(df[col])]:
            df[column] = df[column].replace('нет', 'NaN')
            df[column] = df[column].fillna('NaN')
            
        return df
    
    def get_hexagon_stats(self) -> pd.DataFrame:
        """
        Get statistics calculated for H3 hexagons.
        
        Returns:
            DataFrame containing hexagon statistics.
        """
        if not self.use_hex_features:
            raise ValueError("Hex features were not calculated (use_hex_features=False)")
        if self.hex_stats is None:
            raise ValueError("Hex stats not available. Call prepare_for_model() first.")
        return self.hex_stats.copy()

    
    def get_hexagon_geojson(self) -> Dict:
        """
        Generate GeoJSON for visualizing hexagons.
        
        Returns:
            Dictionary containing GeoJSON data.
        """
        if not self.use_hex_features or self.hex_stats is None:
            raise ValueError("Hex features not available")
        
        features = []
        for hex_id, stats in self.hex_stats.iterrows():
            try:
                boundary = h3.cell_to_boundary(hex_id, geo_json=True)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [boundary]
                    },
                    "properties": stats.to_dict()
                })
            except Exception as e:
                continue
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
        
    def show_hexagons(df):
        """
        Visualize hexagons on a Folium map.
        
        Args:
            df: DataFrame containing hexagon data.
            
        Returns:
            Folium map object.
        """
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)
        for hex_id in df['hex_id'].unique():
            boundary = h3.cell_to_boundary(hex_id)
            folium.Polygon(
                locations=boundary,
                fill=True,
                fill_opacity=0.5,
                popup=f"Объектов: {len(df[df['hex_id']==hex_id])}"
            ).add_to(m)
        return m
    
    def auto_select_resolution(df, target_samples=5):
        """
        Automatically select H3 resolution based on target sample size.
        
        Args:
            df: DataFrame containing coordinates.
            target_samples: Minimum number of samples per hexagon.
            
        Returns:
            Selected H3 resolution level.
        """
        for res in range(10, 7, -1):
            hex_ids = df.apply(lambda r: h3.latlng_to_cell(r['latitude'], r['longitude'], res), axis=1)
            if hex_ids.value_counts().min() >= target_samples:
                return res
        return 8
    

    def reverse_categorical_replacements_and_convertations(self, df):
        """
        Reverse categorical transformations applied to the DataFrame.
        
        Args:
            df: DataFrame with transformed categorical columns.
            
        Returns:
            DataFrame with original categorical values restored.
        """
        if self.label_encoders:
            for col, label_encoder in self.label_encoders.items():
                if col in [col.split('_')[0] for col in df.columns if col in self.object_columns]:
                    df[col] = label_encoder.inverse_transform(df[col])
                    
                    if col in self.transformations.keys():
                        reverse_mapping = self.transformations[col]
                        df[col] = df[col].map(reverse_mapping).fillna(df[col])
    
        for col in df.columns:
            if '_' in col and col in self.object_columns:
                original_col = col.split('_')[0]
                
                if original_col in self.columns_with_commas:
                    continue
                
                if original_col not in df.columns:
                    df[original_col] = None
                
                for _, row in df.iterrows():
                    if row[col] == 1:
                        df.at[_, original_col] = col.split('_', 1)[1]
                
                df.drop(columns=[col], inplace=True)
    
        df[[col for col in self.object_columns if col not in self.columns_with_commas and col in df.columns]] = df[
            [col for col in self.object_columns if col not in self.columns_with_commas and col in df.columns]
        ].fillna('NaN')
    
        return df
    
    def reverse_normalization_logtransformation(self, df, norm_needed = True, log_needed = True):
        """
        Reverse log transformation and normalization applied to the DataFrame.
        
        Args:
            df: DataFrame with transformed values.
            norm_needed: Whether normalization was applied.
            log_needed: Whether log transformation was applied.
            
        Returns:
            DataFrame with original scale values restored.
        """
        print(norm_needed, log_needed)
        if norm_needed:
            if self.scaler is None:
                raise ValueError("Scaler is not fitted. Call 'normalization_logtransformation' first.")
            
            columns_to_normalize = ['houseArea', 'landArea', 'distanceFromMkad', 'year', 'distanceToCityKm', 'price']
            df[columns_to_normalize] = self.scaler.inverse_transform(df[columns_to_normalize])

            if self.lat_long_scaler is None:
                raise ValueError("Latitude/Longitude scaler is not fitted. Call 'normalization_logtransformation' first.")
            df[['latitude', 'longitude']] = self.lat_long_scaler.inverse_transform(df[['latitude', 'longitude']])
        if log_needed:
            df['houseArea'] = np.expm1(df['houseArea'])
            df['landArea'] = np.expm1(df['landArea'])
            df['distanceFromMkad'] = np.expm1(df['distanceFromMkad']) - 1
            df['distanceToCityKm'] = np.expm1(df['distanceToCityKm']) - 1
            df['price'] = np.expm1(df['price'])

        return df
```
