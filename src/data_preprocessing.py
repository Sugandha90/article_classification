# preprocesses the data
import warnings

import pandas as pd

from utils import (fix_na_stats, remove_similar_category, select_subset, encode_category_to_numeric, process_date,
                   clean_author, process_text, process_urls, remove_redundant_columns)

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self, file_path, select_subset_flag, mode):
        self.file_path = file_path
        self.select_subset_flag = select_subset_flag
        self.mode = mode

    def get_dataframe(self):
        return pd.read_json(self.file_path, lines=True)

    def process_data(self):
        df = self.get_dataframe()
        print('\nPreprocessing of data starts:')
        if self.mode == 'train':
            df = fix_na_stats(df)
            df = remove_similar_category(df)
            if self.select_subset_flag is True:
                df = select_subset(df)
            df = encode_category_to_numeric(df, column_name='category')
            df = process_date(df, column_name='date')
            df = process_urls(df, column_name='link')
            df = clean_author(df, column_name='authors')
            df = process_text(df, columns=['headline', 'authors'])
            df = remove_redundant_columns(df)
            df.reset_index(drop=True, inplace=True)  # reset the index

        if self.mode == 'inference':
            df.drop(columns=['category'], inplace=True)
            df = fix_na_stats(df)
            df = process_date(df, column_name='date')
            df = process_urls(df, column_name='link')
            df = clean_author(df, column_name='authors')
            df = process_text(df, columns=['headline', 'authors'])
            df.reset_index(drop=True, inplace=True)  # reset the index

        return df
