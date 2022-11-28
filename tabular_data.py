import pandas as pd
from pandasgui import show

def load_raw_data(path: str = "C:\\Users\\jared\\AiCore\\DS_Airbnb\\AirbnbDataSci\\structured\\AirBnbData.csv"):
    '''
    Loads the data from file path

    Parameters:
    ----------
    path: str
        file path of data
    
    '''
    raw_df = pd.read_csv(path)
    return raw_df

def remove_rows_with_missing_ratings(input_df, column: str):
    '''
    Removes any rows with NaN values in Value_rate by only keeping the rows with values with the notna() module

    Parameters:
    ----------
    input_df: dataframe
        Dataframe of raw data
    column: str
        Name of column to identify NaN values in rows
    
    Returns:
    --------
    df: dataframe
        Dataframe of edited data
    '''
    df = input_df[input_df[column].notna()]
    return df

def combine_description_strings(df, string_column: str):
    '''
    Removes any rows with NaN in Description column. Removes 'About this space' prefix. 
    Then concatenates the multiple strings into 1 string

    Returns:
    --------
    df: dataframe
        Dataframe of edited data
    '''
    df = df[df[string_column].notna()]
    df[string_column] = df[string_column].replace({'About this space': '', '"': ''}, regex=True)

    def fix_lists(df):
        return df.replace('[', '').replace(']', '').replace(',', ' ').replace("'", '').split()

    df[string_column] = df[string_column].apply(fix_lists) 
    df[string_column] = df[string_column].apply(lambda x: ' '.join(x))
    df[string_column] = df[string_column].apply(lambda x: '"' + str(x) + '"')

    return df

def set_default_feature_values(df, cols_for_default_value: list, number: int = 1):
    '''
    Sets the NaN values in given column to a specified number(default=1).

    Parameters:
    ----------
    df: dataframe
        Dataframe of raw data
    column: str
        String of the column title
    number: int
        Number to set any NaN values to
    
    Returns:
    --------
    df: dataframe
        Dataframe of edited data
    '''
    for column in cols_for_default_value:
        df[column] = df[column].fillna(number)
    return df

def load_airbnb(label: str = "Price_Night", str_cols: list = ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"]):
    '''
    Function calls clean_tabular_data() then resets the index of df.
    It then makes two new dfs: one for labels and one for the cols to be removed from original df.
    These are taken in as parameters label, and str_cols.
    By default it removes the str cols but can be adapted otherwise..
    It returns only the feature and label dfs as a tuple.

    Parameters:
    ----------
    label: str
        Name of the label column
    str_cols: list of str
        Names of the columns to remove from the dataframe
    
    Returns:
    --------
    modified_dfs: tuple
        Tuple of (features, labels)
     '''
    df = clean_tabular_data()
    df = df.reset_index(drop=True)
    labels = pd.DataFrame()
    label_to_predict = df.pop(label)
    labels.insert(0, label, label_to_predict)

    text_data = pd.DataFrame()
    col_loc = 0
    for col in str_cols:
        move_cols = df.pop(col)
        text_data.insert(col_loc, col, move_cols)
        col_loc += 1
    features = df
    # show(features)
    # show(text_data)
    # show(labels)
    modified_dfs = (features, labels)
    return modified_dfs
        
def clean_tabular_data(raw_data_file: str = "C:\\Users\\jared\\AiCore\\DS_Airbnb\\AirbnbDataSci\\structured\\AirBnbData.csv"):
    '''
    Calls several other functions to clean the data.

    Returns:
    --------
    edit_str_df: Dataframe
        Cleaned dataframe with rows missing 'Value_rate' removed, rows with NaN set to 1 in ["guests", "beds", "bathrooms", "bedrooms"], 
        and the string descriptions concatenated
    '''
    raw_df = load_raw_data(raw_data_file)
    remove_rows_df = remove_rows_with_missing_ratings(raw_df, 'Value_rate')
    #show(remove_rows_df)
    edit_nan_values_df = set_default_feature_values(remove_rows_df, ["guests", "beds", "bathrooms", "bedrooms"])
    #show(edit_nan_values_df)
    edit_str_df = combine_description_strings(edit_nan_values_df, 'Description')
    #show(edit_str_df)
    return edit_str_df

if __name__ == "__main__":
    load_airbnb()
