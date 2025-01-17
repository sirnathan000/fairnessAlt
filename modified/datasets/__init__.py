
from utils.types import Dataset

def get_dataset_params(name):
    if name == Dataset.ADULT:
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]
        target_var = 'salary-class'
        IS_CAT = [True, False, True, True, True, True, True, True]
        max_numeric = {"age": 50.5}
    elif name == Dataset.ADULT2:
        #todo: add the necessary information to make the ratio work alternatively run the ratio here and pass it allong (dumb thing to do but hey if it works it works)
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]
        target_var = 'salary-class'
        IS_CAT = [True, False, True, True, True, True, True, True]
        max_numeric = {"age": 50.5}
    elif name == Dataset.CMC:
        QI_INDEX = [1, 2, 4]
        target_var = 'method'
        IS_CAT = [False, True, False]
        max_numeric = {"age": 32.5, "children": 8}
    elif name == Dataset.MGM:
        QI_INDEX = [1, 2, 3, 4, 5]
        target_var = 'severity'
        IS_CAT = [True, False, True, True, True]
        max_numeric = {"age": 50.5}
    elif name == Dataset.CAHOUSING:
        QI_INDEX = [1, 2, 3, 8, 9]
        target_var = 'ocean_proximity'
        IS_CAT = [False, False, False, False, False]
        max_numeric = {"latitude": 119.33, "longitude": 37.245, "housing_median_age": 32.5,
                    "median_house_value": 257500, "median_income": 5.2035}
    elif name == Dataset.INFORMS:
        QI_INDEX = [3, 4, 6, 13, 16]
        target_var = "poverty"
        IS_CAT = [True, True, True, True, False]
        max_numeric = {"DOBMM": None, "DOBYY": None, "RACEX":None, "EDUCYEAR": None, "income": None}
    elif name == Dataset.ITALIA:
        QI_INDEX = [1, 2, 3]
        target_var = "disease"
        IS_CAT = [False, True, False]
        max_numeric = {"age": 50, "city_birth": None, "zip_code":50000}
    elif name == Dataset.FOLKSTABLE:
#QI_index show the actual indices of the quasi-identifiers in the data
        QI_INDEX = [1, 2, 3, 4, 6, 7]
#target_var is the variable we want to predict
        target_var = "PINCP"
#IS_CAT is a list of booleans that show if the quasi-identifier is categorical or not
        IS_CAT = [True, True, True, True, True, True]
#max_numeric is a dictionary that shows the maximum value of the quasi-identifiers
        max_numeric = {}
    else:
        print(f"Not support {name} dataset")
        raise ValueError
    return {
        'qi_index': QI_INDEX,
        'is_category': IS_CAT,
        'target_var': target_var,
        'max_numeric': max_numeric
    }
