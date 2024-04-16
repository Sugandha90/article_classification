# Contain tests for the few functions from src/utils
# TO DO: implement remaining tests
import unittest
import pandas as pd
from src import utils


class DataPreprocessor(unittest.TestCase):

    def setUp(self):
        # # dummy data frame
        self.df = pd.DataFrame({'category': ['A', 'B', 'C', 'D'],
                                'category_labels': [0, 1, 2, 3],
                                'date': [1, 2, 3, 4],
                                'link': ['A', 'B', 'C', 'D'],
                                'domain': ['A', 'B', 'C', 'D'],
                                'day_of_week': [4, 5, 6, 7],
                                'headline': ['This is a headline.', 'Another headline here.', 'One more', ''],
                                'authors': ['Aana Contributor Doe', 'Jane Doe', 'Patil', '']})

    def test_fix_na_stats(self):
        # Call the function to fix NaN values and empty strings
        cleaned_df = utils.fix_na_stats(self.df)

        # Check if NaN values and empty strings are removed
        assert cleaned_df.isna().sum().sum() == 0  # Check if there are no NaN values
        assert (cleaned_df == '').sum().sum() == 0  # Check if there are no empty strings

    def test_remove_redundant_columns(self):
        # Expected result after removing redundant columns
        expected_result = pd.DataFrame({'category': ['A', 'B', 'C', 'D'],
                                        'category_labels': [0, 1, 2, 3], 'day_of_week': [4, 5, 6, 7]})

        # Removing redundant columns
        result = utils.remove_redundant_columns(self.df)

        # Asserting if the result matches the expected result
        assert result.equals(expected_result)

    def test_clean_author(self):
        # Clean the author column
        cleaned_df = utils.clean_author(self.df, column_name='authors')
        cleaned_name = cleaned_df['authors'][0]
        expected_name = self.df['authors'][0]
        # Check if the author names are cleaned correctly
        assert cleaned_name == expected_name


if __name__ == '__main__':
    unittest.main()
