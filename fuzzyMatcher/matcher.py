from __future__ import annotations
import time
from rapidfuzz import fuzz, process
from rapidfuzz.process import cdist
import numpy as np
import dask.dataframe as dd
from pathlib import Path
import pandas as pd
import itertools


def progress_line(i=0, m=100, num=50):
    percentage = i/m
    hash_number = round(percentage*num)
    loadbar = '[' + '#'*hash_number + '-'*(num-hash_number) + ']' + f'{round(percentage*100, 2)}%: {i} of {m}'
    return loadbar


def error_catcher(f):
    """ Decorator for catching errors """
    def catch(f):
        error = None
        result = None
        try:
            result = f
        except Exception as e:
            error = e
        return result if result else print(error)

    return catch(f)


@error_catcher
def read_file(file: Path):
    return dd.read_csv(file)


def check_columns(column: str, columns: list[str], which_df: str):
    if column in columns:
        return True
    else:
        raise f"Columns {column} wasn't found in the {which_df} dataset."


class FuzzyMatcher:

    def __init__(self, lhs: Path, rhs: Path, left_on: str, right_on: str, verbose=False):
        self.lhs = read_file(lhs)
        self.rhs = read_file(rhs)
        self.rhs_path = rhs
        if check_columns(left_on, self.lhs.columns, 'left'):
            self.left_on = left_on
        if check_columns(right_on, self.rhs.columns, 'right'):
            self.right_on = right_on
        self.progress = 0
        self.len_product = 0
        self.verbose = True

    def compare_one(self, t, v) -> tuple | None:
        if self.verbose:
            self.progress += 1
            if self.progress % 100 == 0:
                print('\r' + progress_line(self.progress, self.len_product), end='')
        if all([entity in v for entity in t[0]]):
            return ' '.join(t[0]), v

    def compare(self, product) -> list[tuple | None]:
        return list(dict.fromkeys(
            [self.compare_one(t, v) for t, v in product]))

    def simple_match(self, lhs, rhs):
        # TODO: N^2 Issue
        product = itertools.product(zip(lhs.str.split().values), rhs)
        self.len_product = lhs.shape[0] * rhs.shape[0]
        matches = self.compare(product)
        matches.remove(None)
        matches_df = pd.DataFrame(list(matches), columns=['searched', 'found'])
        rhs['index'] = rhs.index
        return matches_df.merge(rhs, left_on='found', right_on=self.right_on)

    def deep_match_one(self, name, rhs, scorer):
        if self.verbose:
            self.progress += 1
            print('\r' + progress_line(self.progress, self.len_product), end='')
        return process.extractOne(name, rhs, scorer=scorer)

    def deep_match(self, lhs, rhs, scorer=fuzz.token_set_ratio):
        print('Performing fuzzy matching...')
        self.len_product = lhs.shape[0]
        return [self.deep_match_one(name, rhs, scorer) for name in lhs]

    def deep_match_inmomory(self, lhs, rhs, scorer=fuzz.token_set_ratio):
        #TODO: implement the output. For now it's not as interesting, since we would able to use it locally
        print('Performing fuzzy matching...')
        similarity_matrix = cdist(lhs, rhs, scorer=scorer, score_cutoff=70, workers=-1)
        matches = []
        scores = []
        index = []
        for distances in similarity_matrix:
            # Get indices of matches
            indeces = np.argwhere(distances == np.amax(distances)).flatten()
            # Get names from indices
            if indeces:
                matches.append(list(map(rhs.__getitem__, indeces)))
                index.append(indeces)


    def merge(self, deep=False):
        lhs = self.lhs[self.left_on].compute()
        rhs = self.rhs[self.right_on].compute()

        lhs = lhs.str.lower()
        rhs = rhs.str.lower()
        lhs = lhs.dropna()
        rhs = rhs.dropna()
        lhs = lhs.drop_duplicates(keep='first')
        print(f'Shape of LHS: {lhs.shape}')

        if not deep:
            matches = self.simple_match(lhs, rhs)
        else:
            start = time.time()
            matches = self.deep_match(lhs, rhs)
            matches = pd.DataFrame(matches, columns=['rhs_name', 'score', 'rhs_index'])
            print(f'\nTime taken: {round(time.time() - start, 2)} seconds.\n')
            matches['lhs_name'] = lhs.values

        return matches

