from fuzzyMatcher.matcher import FuzzyMatcher
from pathlib import Path
# from IPython.display import display

if __name__ == '__main__':
    root = Path('C:\\Users\\M302242\\PycharmProjects\\FastFuzzyMatching')
    lhs = root.joinpath('dcc_data_set3_participants.csv') # Dataset 87816 x 44 - 10.777 distinct names
    rhs = root.joinpath('d_account_filtered.csv')
    M = FuzzyMatcher(lhs=lhs, rhs=rhs, left_on='Full_name', right_on='Account', verbose=True)
    match = M.merge(deep=True, max_matches=3)
    #display(match)
    match.to_excel('produced_match.xlsx')

    match = M.make_scoring(rules={'HCP Franchise': [('Fertility', 1.5), ('Multi-Franchise', 1)], '%Account_Key': None})

    match.to_excel('produced_match_scored.xlsx')