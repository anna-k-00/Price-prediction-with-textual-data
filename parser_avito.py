from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm.auto import tqdm  
from selenium.webdriver.common.by import By
import time
import concurrent.futures
import random 

# Define property types and highways
property_types = {'dom': 'ASgBAgICAkSUA9AQ2AjOWQ',
                  'dacha': 'ASgBAgICAkSUA9AQ2AjQWQ',
                  'kottedzh': 'ASgBAgICAkSUA9AQ2AjKWQ'}

highways = {
    1: 'Алтуфьевское шоссе',
    2: 'Боровское шоссе',
    3: 'Быковское шоссе',
    4: 'Варшавское шоссе',
    5: 'Волоколамское шоссе',
    6: 'Горьковское шоссе',
    7: 'Дмитровское шоссе',
    8: 'Егорьевское шоссе',
    9: 'Ильинское шоссе',
    10: 'Калужское шоссе',
    11: 'Каширское шоссе',
    12: 'Киевское шоссе',
    13: 'Куркинское шоссе',
    14: 'Ленинградское шоссе',
    15: 'Минское шоссе',
    16: 'Можайское шоссе',
    17: 'Новокаширское шоссе',
    18: 'Новорижское шоссе',
    19: 'Новорязанское шоссе',
    20: 'Новосходненское шоссе',
    21: 'Носовихинское шоссе',
    22: 'Осташковское шоссе',
    23: 'Пятницкое шоссе',
    24: 'Рогачёвское шоссе',
    25: 'Рублёво-Успенское шоссе',
    26: 'Рублёвское шоссе',
    27: 'Рязанское шоссе',
    28: 'Симферопольское шоссе',
    29: 'Сколковское шоссе',
    30: 'Фряновское шоссе',
    31: 'Щёлковское шоссе',
    32: 'Ярославское шоссе',
}

class AvitoParser:
    def __init__(self, 
                 target_types=list(property_types.keys()), 
                 target_highways=list(highways.values()), 
                 path_links='raw_links_house.txt',
                 df_path='data_new_house.csv',
                 drop_prev_files=False,
                 parse_new_links=True):
        """
        Initialize the AvitoParser class with the given parameters.

        :param target_types: List of property types to parse.
        :param target_highways: List of highways to parse.
        :param path_links: Path to the file where raw links will be stored.
        :param df_path: Path to the CSV file where parsed data will be stored.
        :param drop_prev_files: Whether to drop previous files.
        :param parse_new_links: Whether to parse new links.
        """
        self.target_types = target_types
        self.target_highways = target_highways
        self.path_links = path_links
        self.df_path = df_path
        self.drop_prev_files = drop_prev_files
        self.parse_new_links = parse_new_links

        self.property_types = {'dom': 'ASgBAgICAkSUA9AQ2AjOWQ',
                  'dacha': 'ASgBAgICAkSUA9AQ2AjQWQ',
                  'kottedzh': 'ASgBAgICAkSUA9AQ2AjKWQ'}

        self.highways = {
            1: 'Алтуфьевское шоссе',
            2: 'Боровское шоссе',
            3: 'Быковское шоссе',
            4: 'Варшавское шоссе',
            5: 'Волоколамское шоссе',
            6: 'Горьковское шоссе',
            7: 'Дмитровское шоссе',
            8: 'Егорьевское шоссе',
            9: 'Ильинское шоссе',
            10: 'Калужское шоссе',
            11: 'Каширское шоссе',
            12: 'Киевское шоссе',
            13: 'Куркинское шоссе',
            14: 'Ленинградское шоссе',
            15: 'Минское шоссе',
            16: 'Можайское шоссе',
            17: 'Новокаширское шоссе',
            18: 'Новорижское шоссе',
            19: 'Новорязанское шоссе',
            20: 'Новосходненское шоссе',
            21: 'Носовихинское шоссе',
            22: 'Осташковское шоссе',
            23: 'Пятницкое шоссе',
            24: 'Рогачёвское шоссе',
            25: 'Рублёво-Успенское шоссе',
            26: 'Рублёвское шоссе',
            27: 'Рязанское шоссе',
            28: 'Симферопольское шоссе',
            29: 'Сколковское шоссе',
            30: 'Фряновское шоссе',
            31: 'Щёлковское шоссе',
            32: 'Ярославское шоссе',
        }


    def initializer(self, target_types=None, target_highways=None, 
                    path_links=None, df_path=None, 
                    drop_prev_files=None, parse_new_links=None):
        """
        Initialize the parsing process with optional parameters.

        :param target_types: List of property types to parse.
        :param target_highways: List of highways to parse.
        :param path_links: Path to the file where raw links will be stored.
        :param df_path: Path to the CSV file where parsed data will be stored.
        :param drop_prev_files: Whether to drop previous files.
        :param parse_new_links: Whether to parse new links.
        """
        # Update class attributes if new values are provided
        if target_types is not None:
            self.target_types = target_types
        if target_highways is not None:
            self.target_highways = target_highways
        if path_links is not None:
            self.path_links = path_links
        if df_path is not None:
            self.df_path = df_path
        if drop_prev_files is not None:
            self.drop_prev_files = drop_prev_files
        if parse_new_links is not None:
            self.parse_new_links = parse_new_links  # Ensure this is updated

        # Debugging: Print the updated attributes
        print(f"Updated attributes: target_types={self.target_types}, parse_new_links={self.parse_new_links}")

        # Generate tuples of property types and highways
        tuples = []
        hw_numbers = [k for k, v in highways.items() if v in self.target_highways]
        for type in self.target_types:
            for highway in hw_numbers:
                tuples.append((type, highway))

        # Start the parsing process
        self.get_dist_ads_urls(tuples)

    def get_dist_ads_urls(self, tuples):
        """
        Get the URLs of the ads based on the tuples of property types and highways.

        :param tuples: List of tuples containing property types and highways.
        """
        counter_ads = 0
        chrome_options = Options()
        prefs = {
            "profile.managed_default_content_settings.images": 2,  # 2 means block images
            "profile.default_content_setting_values.images": 2,    # 2 means block images
        }
        chrome_options.add_experimental_option("prefs", prefs)

        if self.drop_prev_files:
            if os.path.exists(self.path_links):
                os.remove(self.path_links)
            if os.path.exists(self.df_path):
                os.remove(self.df_path)

        driver = webdriver.Chrome(service=Service(), options=chrome_options)
        print(self.parse_new_links)
        if self.parse_new_links:
            print('getting urls')
            for type, highway in tqdm(tuples, desc="Processing highways", unit="highway"):
                try:
                    url = f'https://www.avito.ru/moskovskaya_oblast/doma_dachi_kottedzhi/prodam/{type}-{property_types[type]}?context=H4sIAAAAAAAA_wEjANz_YToxOntzOjg6ImZyb21QYWdlIjtzOjc6ImNhdGFsb2ciO312FITcIwAAAA&road={highway}'
                    driver.get(url)
                    ads_count = int(driver.find_element(by=By.XPATH, value="//span[@data-marker='page-title/count']").text.replace(' ', ''))
                    counter_ads += ads_count
                    if ads_count % 50 > 0:
                        page_count = (ads_count // 50) + 1
                    else:
                        page_count = ads_count // 50

                    for page in range(1, page_count + 1):
                        if page == 1:
                            ads_elements = driver.find_elements(by=By.XPATH, value='//a[@data-marker="item-title"]')
                        else:
                            driver.get(f"https://www.avito.ru/moskovskaya_oblast/doma_dachi_kottedzhi/prodam/{type}-{property_types[type]}?context=H4sIAAAAAAAA_wEjANz_YToxOntzOjg6ImZyb21QYWdlIjtzOjc6ImNhdGFsb2ciO312FITcIwAAAA&p={page}&road={highway}")
                            ads_elements = driver.find_elements(by=By.XPATH, value='//a[@data-marker="item-title"]')

                        with open(self.path_links, mode='a') as f:
                            for ad in ads_elements:
                                link = ad.get_attribute("href")
                                if link:
                                    f.write(link + '\n')
                                print(link)
                except Exception:
                    continue
            driver.quit()
            # print('Total number of ads to parse: ', counter_ads)

        # Use ThreadPoolExecutor to parallelize the parsing of ads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_items_characteristics, self.path_links, self.df_path) for _ in range(10)]  # Adjust the number of threads as needed
            for future in concurrent.futures.as_completed(futures):
                parsed, closed = future.result()
                while parsed + closed < counter_ads:
                    time.sleep(60)
                    parsed, closed = self.get_items_characteristics(self.path_links, self.df_path)

    def get_items_characteristics(self, path_links, df_path):
        print('getting chracteristics')
        """
        Parse the characteristics of the ads from the given URLs.

        :param path_links: Path to the file containing the URLs of the ads.
        :param df_path: Path to the CSV file where parsed data will be stored.
        :return: Tuple containing the number of parsed and closed ads.
        """
        chrome_options = Options()
        prefs = {
            "profile.managed_default_content_settings.images": 2,  # 2 means block images
            "profile.default_content_setting_values.images": 2,    # 2 means block images
        }
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(service=Service(), options=chrome_options)

        parsed_data = []
        failed_links = []
        closed_list = []

        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            df = pd.DataFrame(columns=['id'])

        with open(path_links, 'r') as fp:
            urls = fp.readlines()
        
        random.shuffle(urls)  
        
        for url in tqdm(urls):
            url = url.strip()
            if not url:
                continue

            id = url.split('/doma_dachi_kottedzhi/')[1].split('?context')[0]

            if id in df['id'].values or id in closed_list:
                continue

            try:
                # print(id, 'started')
                driver.get(url)
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                chars = {}
                for ultag in soup.find_all('ul', {'class': 'params-paramsList-_awNW'}):
                    for litag in ultag.find_all('li'):
                        l = litag.text.split(': ')
                        chars[l[0]] = l[1].replace('\xa0', '').replace('сот.', '').replace('м²', '').replace('км', '')

                chars['address'] = soup.find(itemprop="address").text
                chars['price'] = soup.find("span", itemprop="price").text.replace('\xa0', '').replace('₽', '')
                chars['description'] = soup.find('div', {'itemprop': 'description'}).text
                chars['seller'] = soup.find('div', {'data-marker': 'seller-info/label'}).text
                chars['category'] = soup.find_all(itemprop="name")[4].text

                try:
                    chars['lat'] = soup.find(class_='style-item-map-wrapper-ElFsX style-expanded-x335n').attrs['data-map-lat']
                    chars['lon'] = soup.find(class_='style-item-map-wrapper-ElFsX style-expanded-x335n').attrs['data-map-lon']
                except Exception:
                    failed_links.append(url)
                    continue

                chars['id'] = id
                df = pd.concat([df, pd.DataFrame([chars])], ignore_index=True)
                df.to_csv(df_path, index=False)
                parsed_data.append(id)
                # print(id, 'parsed')
                time.sleep(5)
            except Exception as e:
                try:
                    driver.get(url)
                    html = driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    closed = soup.find_all(class_="closed-warning-block-_5cSD")
                    if len(closed) > 0:
                        closed_list.append(id)
                        # print(id, 'closed')
                        time.sleep(5)
                    else:
                        failed_links.append(url)
                        time.sleep(5)
                except Exception as e:
                    failed_links.append(url)
                    time.sleep(5)

        driver.quit()

        if failed_links:
            with open('failed_links.txt', 'w') as f:
                for link in failed_links:
                    f.write(f'{link}\n')
            print('Number of failed adds: ', len(failed_links))
            print('Failed links saved to failed_links.txt')

        if closed_list:
            with open('closed_list.txt', 'w') as f:
                for item in closed_list:
                    f.write(f'{item}\n')
            print('Number of closed adds: ', len(closed_list))
            print('Closed list saved to closed_list.txt')

        print('Number of parsed adds: ', len(parsed_data))
        return [len(parsed_data), len(closed_list)]    
