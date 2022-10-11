import pandas as pd
from pandasgui import show

class TabData:
    '''
    A class to clean and prepare raw tabular data loaded in from a file.

    Attributes:
    ----------
    df: dataframe
        Dataframe of raw data loaded in from file
    
    Methods:
    -------
    remove_rows_with_missing_ratings()
        Removes any rows with NaN values in given column
    combine_description_strings()
        Removes any rows with NaN values in description column, and concatenates multiple strings into 1 string.
    set_default_feature_values()
        Replaces NaN values with int in given column
    '''
    def __init__(self, path: str):
    #def load_raw_data(path: str = "C:\\Users\\jared\\AiCore\\DS_Airbnb\\AirbnbDataSci\\structured\\AirBnbData.csv"):
        '''
        Loads the data from file path

        Parameters:
        ----------
        path: str
            file path of data
        
        '''
        self.df = pd.read_csv(path)
        #return df

    def remove_rows_with_missing_ratings(self, column: str):
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
        self.df = self.df[self.df[column].notna()]
        return self.df

    def combine_description_strings(self):
        '''
        Removes any rows with NaN in Description column. Removes 'About this space' prefix. 
        Then concatenates the multiple strings into 1 string

        Returns:
        --------
        df: dataframe
            Dataframe of edited data
        '''
        self.df = self.df[self.df['Description'].notna()]
        self.df['Description'] = self.df['Description'].replace({'About this space': '', '"': ''}, regex=True)

        def fix_lists(df):
            return df.replace('[', '').replace(']', '').replace(',', ' ').replace("'", '').split()

        self.df['Description'] = self.df['Description'].apply(fix_lists) 
        self.df['Description'] = self.df['Description'].apply(lambda x: ' '.join(x))
        self.df['Description'] = self.df['Description'].apply(lambda x: '"' + str(x) + '"')

        return self.df

    def set_default_feature_values(self, column: str, number: int = 1):
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
        self.df[column] = self.df[column].fillna(number)
        return self.df

    def load_airbnb(self, label: str, str_cols: list):
        #load df, remove string columns, remove 1 input column and make that into a seperate df called labels
        
        labels = pd.DataFrame()
        label_to_predict = self.df.pop(label)
        labels.insert(0, label, label_to_predict)

        # Move text data to a seperate df
        text_data = pd.DataFrame()
        col_loc = 0
        for col in str_cols:
            move_cols = self.df.pop(col)
            text_data.insert(col_loc, col, move_cols)
            col_loc += 1
        features = self.df
        my_tuple = (features, labels)
        return my_tuple
        
def clean_tabular_data():
    tab_data = TabData("C:\\Users\\jared\\AiCore\\DS_Airbnb\\AirbnbDataSci\\structured\\AirBnbData.csv")
    #df = t.load_raw_data()
    tab_data.remove_rows_with_missing_ratings('Value_rate')
    cols_for_default_value = ["guests", "beds", "bathrooms", "bedrooms"]
    for column in cols_for_default_value:
        tab_data.set_default_feature_values(column)
    tab_data.combine_description_strings()
    show(tab_data.df)
    tab_data.load_airbnb("Price_Night", ["ID", "Category", "Title", "Description", "url"])

if __name__ == "__main__":
    #p = TabData()
    #df = p.load_raw_data()
    clean_tabular_data()

