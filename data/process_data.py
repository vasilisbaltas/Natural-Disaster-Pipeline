import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Reads in the two datasets and returns a dataframe of the joined datasets and categories dataset '''

    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates()

    categories = pd.read_csv(categories_filepath)
    categories = categories.drop_duplicates()

    df = messages.merge(categories, how='inner', on=['id'])

    return df, categories


def clean_data(df, categories):
    """ Cleans the combined dataframe  so that it contains the messages with their corresponding category

    :param df:            input combined dataframe(df)
    :param categories:    categories dataframe(df)
    :return:              cleaned dataframe(df)
    """

    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    category_colnames = row.str[:-2]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """ Reads in a dataframe and stores it in a database

    :param df:                     input dataframe(df
    :param database_filename:      path for database
    :return:                       None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Emergency_Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()


