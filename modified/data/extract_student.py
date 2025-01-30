from ucimlrepo import fetch_ucirepo
import pandas as pd
  
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 


# data (as pandas dataframes) 
x = student_performance.data.features
y = student_performance.data.targets 
  
# metadata 
data = pd.concat([x, y], axis=1)


#the following are columns are given in the dataset:
# ['school'1, 'sex'2, 'age'3, 'address'4, 'famsize'5, 'Pstatus'6, 'Medu'7, 'Fedu'8,
#        'Mjob'9, 'Fjob'10, 'reason'11, 'guardian'12, 'traveltime'13, 'studytime'14,
#        'failures'15, 'schoolsup'16, 'famsup'17, 'paid'18, 'activities'19, 'nursery'20,
#        'higher'21, 'internet'22, 'romantic'23, 'famrel'24, 'freetime'25, 'goout'26, 'Dalc'27,
#        'Walc'28, 'health'29, 'absences'30, 'G1'31, 'G2'32, 'G3'33]
#
# of these the following have been deamed as sensitive attributes:
# 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus'6, 'Medu', 'Fedu', Mjob9, Fjob, guardian, traveltime12(13), actvities, nursery, higher, romantic16, health
# and the following are protected atributes (these correspond with the folkstable dataset (adult dataset))
# sex and age (race not taken into consideration as it is not available) however to subsitute this Age was selected as specificlly the age and sex of a student should be protected attributes.

# the overal overview of the meaning for the given hierarchies can be found at https://archive.ics.uci.edu/dataset/320/student+performance

#the commented out code gives an overview of what the data looks like and what the unique values are for the given columns, these values were the used for the hierarchies
#data_sorted = sorted(data['traveltime'].unique().tolist())

#for data in data_sorted:
#    print(f"{data};")


#to have the outcome traget (G3) fit the necessary binary of True or false the following conversion had to take place. to convert the numerical to a true of false the following link was used:
#https://www.rug.nl/feb/education/exchange/incoming/gradeconversionfeb.pdf
#as the data is for purtugal a 10 will be considerd a passing grade so a True and a 9 or lower will be considered a False

data['G3'] = data['G3'].apply(lambda x: True if x >= 10 else False)

# the ID also had to be appended to the rows.
data.insert(0, 'ID', range(1, 1 + len(data)))



# here the data is extracted and converted to a csv for processing in later steps
data.to_csv('student/student.csv',sep = ';', index=False)

