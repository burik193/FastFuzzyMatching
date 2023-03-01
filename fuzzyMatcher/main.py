from fuzzyMatcher.matcher import FuzzyMatcher
import os
from pathlib import Path
# from IPython.display import display

# Here is an example of how to use FuzzyMatcher

if __name__ == '__main__':
    root = Path(os.getcwd())
    lhs = root.joinpath('path_to_left_df') # Dataset 87816 x 44 - 10.777 distinct names
    rhs = root.joinpath('path_to_right_df')
    M = FuzzyMatcher(lhs=lhs, rhs=rhs, left_on='col_name_on_lhs', right_on='col_name_on_rhs', verbose=True)
    match = M.merge(deep=True, max_matches=3)
    #display(match)
    match.to_excel('produced_match.xlsx')

    match = M.make_scoring(rules={'categorical_column_name': [('value_1', 1.5), ('value_2', 1)], 'value_3': None})

    match.to_excel('produced_match_scored.xlsx')
