from folktables import ACSDataSource, ACSIncome
import pandas as pd
#here all the data form the folkstable income is retrived. income is based on the us adult senses data. this one specificlly for the year 2018 in california.
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)

#print(ca_data)

#here the data is transformed into numerical values
ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)
#TODO need to combine the features and labels into one csv *labels need to be at end
#TODO need to add id to beging of data with unique number for each line

combined = pd.concat([ca_features, ca_labels], axis=1)
combined.insert(0, 'ID', range(1, 1 + len(combined)))
#here the overal data is exported to the corresponding
combined.to_csv('folkstable/folkstable_alt.csv', sep = ';', index=False)
#ca_labels.to_csv('folkstable/folkstable_train.txt', index=False)
