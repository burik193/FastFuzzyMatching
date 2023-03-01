from __future__ import annotations
import time
from rapidfuzz import fuzz, process
from rapidfuzz.process import cdist
import numpy as np
import dask.dataframe as dd
from pathlib import Path
import pandas as pd
import itertools
from collections import ChainMap
from typing import List, Tuple


def progress_line(i: int = 0, m: int = 100, num: int = 50) -> str:
    """
    Returns a progress bar as a string based on the given input parameters.

    Parameters:
    i (int): The current progress index. Default value is 0.
    m (int): The maximum progress index. Default value is 100.
    num (int): The number of characters to use for the progress bar. Default value is 50.

    Returns:
    str: A string representing the progress bar.

    Example:
    >>> progress_line(30, 100, 20)
    '[####----------------------]30.0%: 30 of 100'
    """
    percentage = i/m
    hash_number = round(percentage*num)
    loadbar = '[' + '#'*hash_number + '-'*(num-hash_number) + ']' + f'{round(percentage*100, 2)}%: {i} of {m}'
    return loadbar


def error_catcher(f):
    """
    Decorator for catching errors.

    Parameters:
    f (function): The function to decorate.

    Returns:
    function: The decorated function.

    Example:
    >>> @error_catcher
    >>> def my_func():
    >>>     print(1/0) # This will raise a ZeroDivisionError
    >>> my_func() # This will print the error message
    """
    def catch(f):
        error = None
        result = None
        try:
            result = f
        except Exception as e:
            error = e
        return result if result else print(error)

    return catch(f)


def flatten(l: List[List | Tuple]) -> List:
    """
    Flattens a list of lists or tuples.

    Parameters:
    l (List[List | Tuple]): A list of lists or tuples.

    Returns:
    List: A flattened list.

    Example:
    >>> flatten([[1,2,3], [4,5], [6,7,8,9]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return [item for sublist in l for item in sublist]


@error_catcher
def read_file(file: Path):
    """
    Reads a CSV file using Dask and catches any errors.

    Parameters:
    file (Path): The path to the CSV file.

    Returns:
    Dask DataFrame: The contents of the CSV file as a Dask DataFrame.

    Example:
    >>> my_dataframe = read_file("data.csv")
    """
    return dd.read_csv(file)


def check_columns(column: str, columns: List[str], which_df: str) -> bool:
    """
    Checks if a given column is in a list of columns.

    Parameters:
    column (str): The column to check.
    columns (List[str]): The list of columns to check against.
    which_df (str): The name of the dataset being checked.

    Returns:
    bool: True if the column is in the list of columns, False otherwise.

    Example:
    >>> check_columns("age", ["name", "age", "gender"], "my_dataset")
    True
    """
    if column in columns:
        return True
    else:
        raise f"Columns {column} wasn't found in the {which_df} dataset."

def normalized(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """
    Returns a normalized version of a numpy array along a specified axis.

    Args:
    - a (np.ndarray): The numpy array to normalize.
    - axis (int): The axis along which to normalize the array. Defaults to -1.
    - order (int): The normalization order (e.g., L1, L2). Defaults to 2.

    Returns:
    - np.ndarray: A normalized version of the input numpy array.

    Raises:
    - ValueError: If the input array is not a numpy array.
    """
    if not isinstance(a, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Compute the L2 norm of the array along the specified axis
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    
    # If the norm is zero, replace it with one to avoid division by zero
    l2[l2 == 0] = 1
    
    # Divide the array by the norm along the specified axis
    return a / np.expand_dims(l2, axis)


class FuzzyMatcher:
    """
    A class for performing fuzzy matching on two datasets.

    Parameters:
        lhs (Path): The path to the first dataset.
        rhs (Path): The path to the second dataset.
        left_on (str): The name of the column to use as the key in the first dataset.
        right_on (str): The name of the column to use as the key in the second dataset.
        verbose (bool): Whether to print progress updates during the comparison process. Default is False.

    Attributes:
        lhs (DataFrame): The first dataset loaded as a pandas DataFrame.
        rhs (DataFrame): The second dataset loaded as a pandas DataFrame.
        rhs_path (Path): The path to the second dataset.
        left_on (str): The name of the column to use as the key in the first dataset.
        right_on (str): The name of the column to use as the key in the second dataset.
        progress (int): The current progress of the comparison process.
        len_product (int): The total number of comparisons to make.
        verbose (bool): Whether to print progress updates during the comparison process.
        matches (DataFrame): The matches found during the comparison process.

    Methods:
        compare_one(t, v) -> tuple | None:
            Compare a tuple `t` and a row of the second dataset `v`.
            If all entities in `t[0]` are in `v`, return a tuple of the form (`t[0]`, `v`).
            Otherwise, return None.

        compare(product) -> list[tuple | None]:
            Compare all tuples in `product` with all rows in the second dataset.
            Return a list of the tuples that match.

    """
    
    def __init__(self, lhs: Path, rhs: Path, left_on: str, right_on: str, verbose=False):
        """
        Initialize a new instance of the FuzzyMatcher class.

        Args:
            lhs (Path): The path to the first dataset.
            rhs (Path): The path to the second dataset.
            left_on (str): The name of the column to use as the key in the first dataset.
            right_on (str): The name of the column to use as the key in the second dataset.
            verbose (bool, optional): Whether to print progress updates during the comparison process. Default is False.

        Raises:
            Exception: If `left_on` is not found in the columns of the first dataset.
            Exception: If `right_on` is not found in the columns of the second dataset.

        Returns:
            None
        """
        self.lhs = read_file(lhs)
        self.rhs = read_file(rhs)
        self.rhs_path = rhs
        if check_columns(left_on, self.lhs.columns, 'left'):
            self.left_on = left_on
        else:
            raise Exception(f"Columns {left_on} wasn't found in the left dataset.")
        if check_columns(right_on, self.rhs.columns, 'right'):
            self.right_on = right_on
        else:
            raise Exception(f"Columns {right_on} wasn't found in the right dataset.")
        self.progress = 0
        self.len_product = len(self.lhs[['index', left_on]].product(self.rhs[['index', right_on]]))
        self.verbose = verbose
        self.matches = pd.DataFrame()

    def compare_one(self, t, v) -> tuple | None:
    """
    Compare a single row from the left dataframe with a single row from the right dataframe.
    
    Args:
    - self: instance of FuzzyMatcher class
    - t: tuple containing a single row from the left dataframe
    - v: single row from the right dataframe
    
    Returns:
    - A tuple containing the match (if it exists), otherwise None
    """
    # Check if verbose is True and print progress bar
    if self.verbose:
        self.progress += 1
        if (self.progress/self.len_product * 100) % 1 == 0:
            print('\r' + progress_line(self.progress, self.len_product), end='')
    
    # Check if all entities in left dataframe row match with entities in the right dataframe row
    if all([entity in v for entity in t[0]]):
        return ' '.join(t[0]), v

    def compare(self, product) -> list[tuple | None]:
        """
        Compare all rows in the left dataframe with all rows in the right dataframe.

        Args:
        - self: instance of FuzzyMatcher class
        - product: a cartesian product of the left and right dataframes

        Returns:
        - A list of tuples containing matches (if they exist), otherwise None
        """
        return list(dict.fromkeys(
            [self.compare_one(t, v) for t, v in product]))

    def simple_match(self, lhs, rhs):
        """
        Performs a simple match between the left-hand side and right-hand side dataframes.
        Uses N^2 method, which may be slow for large datasets.

        Args:
            self: the FuzzyMatcher instance.
            lhs (pandas.DataFrame): the left-hand side dataframe.
            rhs (pandas.DataFrame): the right-hand side dataframe.

        Returns:
            pandas.DataFrame: a dataframe with the matches, containing 'lhs_name', 'rhs_name', and 'score' columns.
        """
        # TODO: Solve N^2 Issue
        rhs.name = 'rhs_name'
        rhs = pd.DataFrame(rhs)
        product = itertools.product(zip(lhs.str.split().values), rhs['rhs_name'])
        self.len_product = lhs.shape[0] * rhs.shape[0]
        matches = self.compare(product)
        matches.remove(None)
        matches_df = pd.DataFrame(list(matches), columns=['lhs_name', 'rhs_name'])
        rhs['rhs_index'] = list(rhs.index)
        # may be use deep_match with some scoring for score?
        matches_df['score'] = 100
        return matches_df.merge(rhs, on='rhs_name')

    def deep_match_one(self, name, rhs, scorer, max_matches=3):
        """
        Perform fuzzy matching between a name and a dataframe, returning a dictionary with the results.

        Args:
            name (str): Name to match.
            rhs (pd.DataFrame): Dataframe to match with.
            scorer (fuzzywuzzy.process.extract method): A method for comparing two strings that returns an integer score.
                The default method used is 'token_set_ratio' from fuzzywuzzy.
            max_matches (int): Maximum number of matches to return.

        Returns:
            dict: A dictionary containing the matching results.

        Raises:
            Exception: If the maximum number of matches is set to 0.
        """
        if max_matches == 0:
            raise Exception('Number of max. possible matches have to greater than 0.')
        if self.verbose:
            self.progress += 1
            print('\r' + progress_line(self.progress, self.len_product), end='')
        # Perform matching using the chosen scorer and maximum number of matches
        if max_matches == 1:
            return {name: [process.extractOne(name, rhs, scorer=scorer)]}
        else:
            return {name: process.extract(name, rhs, scorer=scorer, limit=max_matches)}


    def deep_match(self, lhs, rhs, scorer=fuzz.token_set_ratio, max_matches=3) -> pd.D:
        """
        Perform fuzzy matching between two dataframes using a deep match approach.

        Args:
            lhs (pd.DataFrame): Left dataframe to be matched.
            rhs (pd.DataFrame): Right dataframe to be matched.
            scorer (fuzzywuzzy.process.extract method): A method for comparing two strings that returns an integer score.
                The default method used is 'token_set_ratio' from fuzzywuzzy.
            max_matches (int): Maximum number of matches to return per name.

        Returns:
            pd.DataFrame: A dataframe containing the matching results, with columns 'lhs_name', 'rhs_name', 'score' and
            'rhs_index'.

        Raises:
            None

        """
        print('Performing fuzzy matching...')
        # Set length of the product to be compared
        self.len_product = lhs.shape[0]
        # Perform deep matching
        result = [self.deep_match_one(name, rhs, scorer, max_matches=max_matches) for name in lhs]
        result = dict(ChainMap(*result))
        data = []
        for name, matches in result.items():
            names = list(map(lambda x: [name] + list(x), matches))
            data.append(pd.DataFrame.from_records(names, columns=['lhs_name', 'rhs_name', 'score', 'rhs_index']))
        # Combine the results of all the matches into a single dataframe
        return pd.concat(data)

    def deep_match_inmemory(self, lhs, rhs, scorer=fuzz.token_set_ratio, num_workers=-1):
        """
        Perform fuzzy matching between two arrays of strings using cdist function.
        :param lhs: numpy array of strings.
        :param rhs: numpy array of strings.
        :param scorer: fuzzy matching scorer. Default is fuzz.token_set_ratio.
        :param num_workers: number of workers for multiprocessing. Default is -1.
        :return: List of matches.
        """
        # Print progress message.
        print('Performing fuzzy matching...')

        # Compute similarity matrix.
        similarity_matrix = cdist(lhs, rhs, scorer=scorer, score_cutoff=70, workers=num_workers)

        # Initialize lists to store matches, scores and indices.
        matches = []
        scores = []
        index = []

        # Loop through each row in the similarity matrix.
        for distances in similarity_matrix:
            # Get indices of matches.
            indeces = np.argwhere(distances == np.amax(distances)).flatten()

            # Get names from indices.
            if indeces is not []:
                matches.append(list(map(rhs.__getitem__, indeces)))
                index.append(indeces)

        # Return the list of matches.
        return matches

    def make_scoring(self, rules: dict):
        """
        Scores the matches according to rules.

        Parameters:
        -----------
        rules : dict
            A dictionary containing the columns to be scored and the corresponding list of scoring rules.

        Returns:
        --------
        rhs : pandas DataFrame
            A Pandas DataFrame containing the matched records and their scores.
        """
        print('Scoring the matches according to rules...')

        # Get columns to be scored
        target_columns = list(rules.keys()) + [self.right_on]

        # Get the corresponding records from the right DataFrame
        rhs = self.rhs[target_columns].compute()
        rhs = rhs.loc[self.matches['rhs_index']]

        # Set initial scores
        rhs['score'], rhs['adj_score'] = self.matches['score'].to_list(), self.matches['score'].to_list()

        # Apply the scoring rules
        for col, rules in rules.items():
            if rules is not None:
                for rule in rules:
                    filter = rhs[col] == rule[0]
                    rhs.loc[filter, 'adj_score'] = rhs.loc[filter, 'adj_score'] * rule[1]

        # Normalize the scores
        rhs['adj_score'] = normalized(rhs['adj_score'].values)[0]
        rhs['adj_score'] = round((rhs['adj_score']/max(rhs['adj_score'])) *100, 2)

        # Add the left-hand side names to the DataFrame
        rhs['lhs_name'] = self.matches['lhs_name'].values

        print('Done.')

        return rhs

    def merge(self, deep=False, in_memory=False, num_workers=4, max_matches=3):
        """
        Performs merging of the two dataframes based on the specified columns using fuzzy matching.

        Args:
            deep (bool): whether or not to perform deep matching (default is False)
            in_memory (bool): whether or not to perform deep matching in-memory (default is False)
            num_workers (int): number of workers to use in parallelization (default is 4)
            max_matches (int): maximum number of matches to find for each record (default is 3)

        Returns:
            pandas DataFrame: the resulting merged dataframe
        """
        # Compute left and right dataframes
        lhs = self.lhs[self.left_on].compute()
        rhs = self.rhs[self.right_on].compute()

        # Preprocessing: lowercase, drop nulls and duplicates
        lhs = lhs.str.lower()
        rhs = rhs.str.lower()
        lhs = lhs.dropna()
        rhs = rhs.dropna()
        lhs = lhs.drop_duplicates(keep='first')

        print(f'Shape of LHS: {lhs.shape}')
        start = time.time()

        # Perform simple or deep matching
        if not deep:
            self.matches = self.simple_match(lhs, rhs)
        else:
            if in_memory:
                matches = self.deep_match_inmemory(lhs, rhs, num_workers=num_workers)
            else:
                self.matches = self.deep_match(lhs, rhs, max_matches=3)

        print(f'\nTime taken: {round(time.time() - start, 2)} seconds.\n')

        return self.matches


#%%
