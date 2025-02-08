# fairness

this repository is a fork from https://github.com/kaylode/k-anonymity which inturn has also forked https://github.com/qiyuangong/Basic_Mondrian among many other repos

this repository has been chosen as it has been used by previous master students and has implemented some logic allowing it to be modified to use alternative datasets more easily then the original

it also contains/uses the following:
https://github.com/socialfoundations/folktables
https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

this project aims to introduce fairnes into basic mondrain. it hopes to do this using the hierarichal settings allowing it to select the most diverse and the most unfair attributes.

## how to use
it is important to know that the method modified_mondrian was used for the generation of datasets in the paper. Hower, i have also added an alt_modified_mondrian that can use columns name from the start

to use the modified_mondrian enter the dir modified and run the following command:
python3 anonymize.py --method=modified_mondrian --k=5 --dataset=student

if you want to use the modified_mondrian with antoher dataset you need to adjust the values in anonymize.py to match the required ones for that dataset.
you also need to adjust the columns attribute in the function AltRatio located in modified\algorithms\modified_mondrian\mondrian.py

to use the alt_modfied_mondrian enter the dir modified and run the following command:
python3 anonymize.py --method=alt_modified_mondrian --k=5 --dataset=student

if you want to switch between datasets you only need to adjust the values in the anonymize.py

## how to add more algorithms
1. to add more algorithms an directory with the same file structer (and __init__.py) needs to be added to run that algorithm
2. after this has been done. the __init__.py in the dir algorithms needs to be appended to match the requirments for the new algorithm.
3. append types.py in modified/utils/types to also include the algorithm
4. append anonymize.py with the required anon.params but also in the assert statement
5. run and test if the new algorithm works

## how to add dataset
1. create a dir in /modified/data with the name of the dataset.
2. add all files of dataset here as well as hierarchies in the an new subdir hierachies.
3. append /modified/utils/types.py with just the line DATASET_NAME = 'dataset_name'  
4. append /modified/utils/__init__.py with the required information for that dataset
5. run and test

## requirments to run
to run this project the following need to be installed using pip.
1. install lightGBM https://github.com/microsoft/LightGBM/tree/master/python-package
2. install af360 https://aif360.readthedocs.io/en/latest/Getting%20Started.html
3. f1 score https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html
4. pandas
5. numpy


