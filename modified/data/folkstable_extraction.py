from folktables import ACSDataSource, ACSIncome
import pandas as pd
#here all the data form the folkstable income is retrived. income is based on the us adult senses data. this one specificlly for the year 2018 in california.
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)


ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)


combined = pd.concat([ca_features, ca_labels], axis=1)
combined.insert(0, 'ID', range(1, 1 + len(combined)))
#here the overal data is exported to the corresponding amount of 30 000  and then exported.
combined = combined.head(30000)
combined.to_csv('folkstable/folkstable.csv', sep = ';', index=False)

