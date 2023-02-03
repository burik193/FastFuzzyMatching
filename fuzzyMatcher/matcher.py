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

def flatten(l: list[list|tuple]) -> list:
    return [item for sublist in l for item in sublist]

@error_catcher
def read_file(file: Path):
    return dd.read_csv(file)


def check_columns(column: str, columns: list[str], which_df: str):
    if column in columns:
        return True
    else:
        raise f"Columns {column} wasn't found in the {which_df} dataset."

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

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
        self.matches = pd.DataFrame()

    def compare_one(self, t, v) -> tuple | None:
        if self.verbose:
            self.progress += 1
            if (self.progress/self.len_product * 100) % 1 == 0:
                print('\r' + progress_line(self.progress, self.len_product), end='')
        if all([entity in v for entity in t[0]]):
            return ' '.join(t[0]), v

    def compare(self, product) -> list[tuple | None]:
        return list(dict.fromkeys(
            [self.compare_one(t, v) for t, v in product]))

    def simple_match(self, lhs, rhs):
        # TODO: N^2 Issue
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
        if max_matches == 0:
            raise 'Number of max. possible matches have to greater than 0.'
        if self.verbose:
            self.progress += 1
            print('\r' + progress_line(self.progress, self.len_product), end='')

        if max_matches == 1:
            return {name: [process.extractOne(name, rhs, scorer=scorer)]}
        else:
            return {name: process.extract(name, rhs, scorer=scorer, limit=max_matches)}

    def deep_match(self, lhs, rhs, scorer=fuzz.token_set_ratio, max_matches=3) -> pd.D:
        print('Performing fuzzy matching...')
        self.len_product = lhs.shape[0]
        result = [self.deep_match_one(name, rhs, scorer, max_matches=max_matches) for name in lhs]
        result = dict(ChainMap(*result))
        data = []
        for name, matches in result.items():
            names = list(map(lambda x: [name] + list(x), matches))
            data.append(pd.DataFrame.from_records(names, columns=['lhs_name', 'rhs_name', 'score', 'rhs_index']))
        return pd.concat(data)

    def deep_match_inmemory(self, lhs, rhs, scorer=fuzz.token_set_ratio, num_workers=-1):
        #TODO: implement the output. For now it's not as interesting, since we would able to use it locally
        print('Performing fuzzy matching...')
        similarity_matrix = cdist(lhs, rhs, scorer=scorer, score_cutoff=70, workers=num_workers)
        matches = []
        scores = []
        index = []
        for distances in similarity_matrix:
            # Get indices of matches
            indeces = np.argwhere(distances == np.amax(distances)).flatten()
            # Get names from indices
            if indeces is not []:
                matches.append(list(map(rhs.__getitem__, indeces)))
                index.append(indeces)

        return matches

    def make_scoring(self, rules: dict):
        print('Scoring the matches according to rules...')
        target_columns = list(rules.keys()) + [self.right_on]
        rhs = self.rhs[target_columns].compute()
        rhs = rhs.loc[self.matches['rhs_index']]
        rhs['score'], rhs['adj_score'] = self.matches['score'].to_list(), self.matches['score'].to_list()
        for col, rules in rules.items():
            if rules is not None:
                for rule in rules:
                    filter = rhs[col] == rule[0]
                    rhs.loc[filter, 'adj_score'] = rhs.loc[filter, 'adj_score'] * rule[1]
        rhs['adj_score'] = normalized(rhs['adj_score'].values)[0]
        rhs['adj_score'] = round((rhs['adj_score']/max(rhs['adj_score'])) *100, 2)
        rhs['lhs_name'] = self.matches['lhs_name'].values
        print('Done.')
        return rhs

    def merge(self, deep=False, in_memory=False, num_workers=4, max_matches=3):
        lhs = self.lhs[self.left_on].compute()
        rhs = self.rhs[self.right_on].compute()

        lhs = lhs.str.lower()
        rhs = rhs.str.lower()
        lhs = lhs.dropna()
        rhs = rhs.dropna()
        lhs = lhs.drop_duplicates(keep='first')
        print(f'Shape of LHS: {lhs.shape}')
        start = time.time()

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
