def num_cat_splitter(df):
    num_list = []
    cat_list = []
    for col in df:
        if df[col].nunique() > 10 and df[col].dtypes != 'O':
            num_list.append(col)
            num = df[num_list]
        elif df[col].nunique() < 10:
            cat_list.append(col)
            cat = df[cat_list]
            
            #cat_nom =cat['league', 'h_a']
        else:
            pass
    
    cat_ord = cat[['year']]
    cat_nom =cat[['league', 'h_a']]
    return num, cat_ord, cat_nom

def min_max(train, test):
    from sklearn.preprocessing import MinMaxScaler 
    transformer = MinMaxScaler().fit(train) #It is important to make sure it only fits the train set
    
    train_normalized = transformer.transform(train)
    test_normalized = transformer.transform(test)
    train_normalized_df = pd.DataFrame(train_normalized, columns=train.columns)
    test_normalized_df = pd.DataFrame(test_normalized, columns=test.columns)
    return train_normalized_df, test_normalized_df

def label_encoder(train, test):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train)
    train_enc = label_encoder.transform(train) 
    test_enc = label_encoder.transform(test)
    train_enc_df = pd.DataFrame(train_enc,columns=train.columns)
    test_enc_df = pd.DataFrame(test_enc, columns=train.columns) 
    return train_enc_df, test_enc_df

def oh_encoder(train, test):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first')# The option drop='first' drops one of the possible values  
    encoder.fit(train)
    train_enc = encoder.transform(train).toarray()
    test_enc = encoder.transform(test).toarray()
    cols = encoder.get_feature_names(input_features=train.columns)
    train_encoded_df = pd.DataFrame(train_enc, columns=cols)
    test_encoded_df = pd.DataFrame(test_enc, columns=cols)
    return train_encoded_df, test_encoded_df

def cleaning_football(com, per_game):
    '''
    This function will clean two datasets and return them. 
    First dataset is for compiled statistics, the second dataset is for per game stats. 
    So please pay attention to the order of datasets when using the function.
    Also this function does not deal with NaN values. For that you may have to check manually.
    The inital datasets had no NaNs and therefore cleanining NaNs not included
    Also if needed, you will have to export file manually
    '''
    com = com.rename({'Unnamed: 0': 'league', 'Unnamed: 1': 'year', 'missed':'against'}, axis=1)
    com = com[com.league != 'RFPL']
    per_game = per_game[per_game.league != 'RFPL']
    com['goals/game'] = com['scored']/understat_com['matches'] 
    com['against/game'] = com['against']/understat_com['matches']
    com.drop_duplicates(keep='first', inplace=True)
    per_game.drop_duplicates(keep='first', inplace=True)
    
    return com, per_game
