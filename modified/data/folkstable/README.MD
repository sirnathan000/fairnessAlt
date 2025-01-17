here all the codes are given these codes are inpart deriffed from 
https://github.com/socialfoundations/folktables/blob/main/examples/encoding_categorical_features.ipynb
and these https://www2.census.gov/programs-surveys/acs/tech_docs/pums/2023ACS_PUMS_User_Guide.pdf
however they also contain the categorie information for generalization purposes

any of the information read was deriffed from the above sources. however recreation was not always possible.
to validate the vlues given below where presenet in the data the read_data.py in the given folder was used.

ACSIncome_categories = {
    #class of worker
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
        10.0: "goverment employee (3.0, 4.0, 5.0)",
        11.0: "private orginization (1.0, 2.0, 6.0, 7.0, 8.0)",
        12.0: "Other (8.0 9.0)",
    },
    #school aka education
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
        25.0: "elementry school (2.00 until 9.0)",
        26.0: "high school (10.0 until 17.0 except 15.0)",
        27.0: "dropped out (15.0, 18.0, 19.0)",
        28.0: "higher education (20.0 until 24.0)",
    },
    #marital status
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
        6.0: "had/has spouse 1.0, 2.0",
        7.0: "no spouse 3.0, 4.0, 5.0 (also 3.0)",
    },
    #sex
    "SEX": {1.0: "Male", 2.0: "Female", 3.0: "other"},
    #race
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
        10.0: "*"
    },
    #age
    "AGEP": {
        101.0: "0 to 4 years",
        102.0: "5 to 9 years",
        103.0: "10 to 14 years",
        104.0: "15 to 19 years",
        105.0: "20 to 24 years",
        106.0: "25 to 29 years",
        107.0: "30 to 34 years",
        108.0: "35 to 39 years",
        109.0: "40 to 44 years",
        110.0: "45 to 49 years",
        111.0: "50 to 54 years",
        112.0: "55 to 59 years",
        113.0: "60 to 64 years",
        114.0: "65 to 69 years",
        115.0: "70 to 74 years",
        116.0: "75 to 79 years",
        117.0: "80 to 84 years",
        118.0: "85 to 89 years",
        119.0: "90 to 94 years",
        120.0: "95 to 99 years",
        121.0: "100 years and over",
    },
    #place of birth
    "POBP":{
    },
    #occupation
    # i was unable to recover the categories for the codes there for it wasn't considerd a quasi feature
    "OCCP':{
    },
    #Relationship to householder: i was unable to recove the exact categories for the codes there for it wasn't considerd a quasi feature
    "RELP": {
    },
    #worked hours per week i was unable to recove the exact categories for the codes there for it wasn't considerd a quasi feature
    "WKHP": {
    },
    #income
    PNICP: {
    "FALSE": "income lower than 50000"
    "TRUE": "income higher than 50000"
    },
}
