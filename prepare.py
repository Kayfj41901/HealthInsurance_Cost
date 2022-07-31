from imports import *


def split_insurance_data(df):
    '''
    This function performs split on zillow data
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test

def select_rfe (X_df, y_df, n = 1, model = LinearRegression(normalize=True), rank = False):
    '''
    Takes in the predictors, the target, and the number of features to select (k) ,
    and returns the names of the top k selected features based on the Recursive Feature Elimination (RFE)
    
    X_df : the predictors
    y_df : the target
    n_features : the number of features to select (k)
    method : LinearRegression, LassoLars, TweedieRegressor
    Example
    select_rfe(X_train_scaled, y_train, 2, LinearRegression())
    '''
    
    rfe = RFE(estimator=model, n_features_to_select= n)
    rfe.fit_transform(X_df, y_df)
    mask = rfe.get_support()
    rfe_feature = X_df.iloc[:,mask].columns.tolist()
    # check if rank=True
    if rank == True:
        # get the ranks
        var_ranks = rfe.ranking_
        # get the variable names
        var_names = X_df.columns.tolist()
        # combine ranks and names into a df for clean viewing
        rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
        # sort the df by rank
        rfe_ranks_df = rfe_ranks_df.sort_values('Rank')
        # print DataFrame of rankings
    return rfe_feature, rfe_ranks_df

def prepare_insurance_for_modeling(df):
    #remove outliers
    df.drop(df[df['bmi'] > 46].index, inplace = True)
    df.drop(df[df['charges'] > 34487].index, inplace = True)
    #map categorical columns to numerical
    df['is_female'] = df.sex.map({'female': 1, 'male': 0})
    df['smoker_encoded'] = df.smoker.map({'yes': 1, 'no': 0})
    df['region_encoded'] = df.region.map({'southeast': 1, 'southwest': 0, 'northeast': 2, 'northwest':3})
    #drop categorical features
    df.drop(columns=['sex', 'smoker', 'region'], inplace=True)
    return df