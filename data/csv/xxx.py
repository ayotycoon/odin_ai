import pandas as pd

# df = pd.read_csv('data/csv/Unmodified Data/Datas.csv')
# x = df.groupby('Category')
# for i,h in x:
#     h.to_csv(f"data/csv/synthetic/brands/sample3/{i or 'unknown'}_j.csv")
# h = df[df["Category"].isna()]
# h.to_csv(f"data/csv/synthetic/brands/sample2/{'unknown'}_j.csv")

df = pd.read_csv('data/csv/Discover-AllAvailable-20250307.csv')
df.drop_duplicates(subset=['Description'], keep='first', inplace= True)
df.to_csv(f"data/csv/Discover-AllAvailable-20250307.csv")
# df.drop(columns=['Description', 'description'], inplace=True)

# def filter_gas_df(cities_df, gas_df):
#     # Create a set of city and state_id values for fast lookup
#     cities_df['x'] = cities_df['city'] + " " + cities_df['state_id']
#     city_state_set = set(cities_df['x'])
#
#     # Define a function to replace matched city-state text in the description
#     def replace_match(description):
#         for part in city_state_set:
#             if part in description:
#                 description = description.replace(part, "")
#         return description
#
#     # Replace matches in the 'Description' column
#     gas_df['Description'] = gas_df['Description'].apply(replace_match)
#
#     return gas_df
#
# cities_df = pd.read_csv('data/csv/synthetic/cityState/uscities.csv')
# gas_df = pd.read_csv('data/csv/Unmodified Data/Master/Auto - Maintenance 2k_MASTER.csv')
#
# x = filter_gas_df(cities_df,gas_df)
# x.to_csv('data/csv/Unmodified Data/Master/Auto - Maintenance 2k_MASTER2.csv')

"""




df = pd.read_csv('data/csv/synthetic/brands/xx.csv')
h = df[df["Category"].isna()]
h.to_csv(f"data/csv/synthetic/brands/{ 'unknown'}_j.csv")


"""


if __name__ == '__main__':
    y = "df"