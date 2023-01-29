from fuzzyMatcher.matcher import FuzzyMatcher
from pathlib import Path
# from IPython.display import display

if __name__ == '__main__':
    root = Path('C:\\Users\\M302242\\PycharmProjects\\Fuzzy_Matching\\CSVs')
    lhs = root.joinpath('dcc_data_set3_participants.csv')
    rhs = root.joinpath('D_Account_Fertility.csv')
    M = FuzzyMatcher(lhs=lhs, rhs=rhs, left_on='Name', right_on='Name', verbose=True)
    match = M.merge(deep=True)
    #display(match)
    match.to_excel('produced_match.xlsx')