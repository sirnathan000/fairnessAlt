# -*- coding: utf-8 -*-

"""
main module of basic Mondrian
"""


import pdb
import time
from functools import cmp_to_key



from tqdm import tqdm
from .models.numrange import NumRange
from .utils.utility import cmp_str

#for ratio
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.sklearn.metrics import disparate_impact_ratio

__DEBUG = False
QI_LEN = 8
SA_INDEX = []
GL_K = 0
RESULT = []
ATT_TREES = []
QI_RANGE = []
IS_CAT = []
GL_L = 0


class Partition(object):

    """Class for Group, which is used to keep records
    Store tree node in instances.
    self.member: records in group
    self.width: width of this partition on each domain. For categoric attribute, it equal
    the number of leaf node, for numeric attribute, it equal to number range
    self.middle: save the generalization result of this partition
    self.allow: 0 donate that not allow to split, 1 donate can be split
    """

    def __init__(self, data, width, middle):
        """
        initialize with data, width and middle
        """
        self.member = list(data)
        self.width = list(width)
        self.middle = list(middle)
        self.allow = [1] * QI_LEN

    def __len__(self):
        """
        return the number of records in partition
        """
        return len(self.member)

def get_normalized_width(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    #print(partition.member[index])
    if IS_CAT[index] is False:
        low = partition.width[index][0]
        high = partition.width[index][1]
        width = float(ATT_TREES[index].sort_value[high]) - float(ATT_TREES[index].sort_value[low])
    else:
        width = partition.width[index]
    return width * 1.0 / QI_RANGE[index]

def altRatio(partition, QID_NAMES, goal, outcome):
#Todo start using this one,
#add if statement and make it 1 (most ideal) if return is nan
#check if return type is float or INT
#combine with normalize width or choose dimension (look back into this) to check combine this with the width for feature selection. DONE
#figure out way to get goal, outcome, and protected down here (i currently can't find where these parameters are passed) DONE


    # need to change this back into pandas dataframe otherwise the aif360 will not work
    # TO CHANGE THIS I NEED TO USE THE partion.member[index] as this is how the data is actually formed
#    data = partition.member[index]
#todo change the columns into the attribute names
    columns = ['sex','age','race','marital-status','education','native-country','workclass','occupation','value','salary-class']
    data = pd.DataFrame(partition.member, columns=columns)



# Clean the dataset
    data.dropna(inplace=True)

    # Convert the goal to binary outcome
#    print("input for goal is ",goal)
#    print("input for outcome is ",outcome)
#    print("input for QUID_Names is ",QID_NAMES)
#    print(data.head())
#    data[goal] = data[goal].apply(lambda x: 1 if x == outcome else 0)

    if outcome not in data[goal].values:
#        print(f"Warning: Outcome value '{outcome}' not found in goal column")
        return 0.00
# Convert the goal to binary outcome
    data[goal] = data[goal].apply(lambda x: 1 if str(x).strip().lower() == str(outcome).strip().lower() else 0)
#    print("data[goal] outputs:", data[goal], "and the count is:", data[goal].value_counts())

    # Encode the protected attribute (e.g., 'race')
    protected_attr = data[QID_NAMES]
    le = LabelEncoder()
    protected_attr = le.fit_transform(protected_attr)

    outcomes = data[goal]

#    print("the outcome var for ratio is:", outcomes)
#    print("the protected var for ratio is:", protected_attr)
#    print("the goal var is:", goal)
#    ratio = 0.50
#    ratio = float(ratio)
#    return ratio
#    the problem with returning nan is here
    if len(outcomes) == 0 or len(protected_attr) == 0:
 #       print("Warning: Empty slices detected in outcomes or protected_attr")
        ratio = 0.00
        ratio = float(ratio)
        return ratio
    # Calculate disparate impact ratio
    try:
        ratio = disparate_impact_ratio(y_true=outcomes, y_pred=outcomes, prot_attr=protected_attr)
        ratio = float(ratio)
        if np.isnan(ratio):
#            print("Warning: Disparate Impact Ratio calculation resulted in nan")
            ratio = 0.00
            ratio = float(ratio)
            return ratio
#        print(f"Disparate Impact Ratio (Race): {ratio:.2f}")
        return ratio
    except Exception as e:
#       print(f"Error calculating Disparate Impact Ratio: {e}")
        ratio = 0.00
        ratio = float(ratio)
        return ratio



""""
this function is not used in the code and might still be usefull lateron
def ratio(data, QI, Protected, goal, outcome):
#TODO make the fucntion work with the variables
#in this function returns the disparte impact ratio of the dataset. the goal must be the the overal target (doesn't have to be the target for the model) (for example income)
#the goal needs to be a binary outcome, the outcome needs to be the favorable outcome (for exmaple above 50K). the QI needs to be the the quasile attribute (for example sex) and Protected needs to be the protected class (for example female)
    data.dropna(inplace=True)

    # Convert income to binary outcome
    data[goal] = data[goal].apply(lambda x: 1 if x == outcome else 0)

    # Protected attribute: Sex
    protected_attr = data[QI].apply(lambda x: 1 if x == Protected else 0)
    outcomes = data[goal]

    # Calculate disparate impact ratio
    ratio = disparate_impact_ratio(y_true=outcomes, y_pred=outcomes, prot_attr=protected_attr)
    return ratio
"""

#commented out the train_model function as it is not used in the code and might still be usefull lateron (i realized i was overthinking the implementation of disparate impact)

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline

#def train_model(data, categorical_features):
#    """
#    Train a machine learning model to predict the sensitive attribute.
#    Returns the predicted values (y_pred) and actual values (A).
#    """
#    # Exclude the sensitive attribute from the features
#    X = np.delete(data, SA_INDEX, axis=1)  # Features
#    A = data[:, SA_INDEX]  # Actual sensitive attribute values
#
#    # Split the data into training and testing sets
#    X_train, X_test, A_train, A_test = train_test_split(X, A, test_size=0.2, random_state=42)#

    # Adjust categorical feature indices after removing the sensitive attribute
#    adjusted_categorical_features = [i if i < SA_INDEX else i - 1 for i in categorical_features]
#
#    # Define the column transformer to handle categorical features
#    preprocessor = ColumnTransformer(
#        transformers=[
#            ('cat', OneHotEncoder(handle_unknown='ignore'), adjusted_categorical_features)
#        ],
#        remainder='passthrough'
#    )

    # Create a pipeline with the preprocessor and the model
#    model = Pipeline(steps=[
#        ('preprocessor', preprocessor),
#        ('classifier', RandomForestClassifier())
#    ])

    # Train the model
#    model.fit(X_train, A_train)

    # Make predictions on the test set
#    y_pred = model.predict(X_test)

#    return y_pred, A_test

def choose_dimension(partition, QID_NAMES, Protected_att, goal, outcome):
    """
    chooss dim with largest normlized Width
    return dim index.
    """
    #TODO add the ratioALT function and make it 1-ratio as the highest width is selected. to ensuer that they never superseed th max of 1 the
    #TODO ratio and the width need to be set to 0.5*ratio + 0.5*width for the ones without a protected attribute it should be 0.5*width to ensure that the protected attribute is always selected first.
    max_width = -1
    max_dim = -1
    for i in range(QI_LEN):
        if partition.allow[i] == 0:
            continue
#todo normWidth for non-protected is two high set to 0.5* so it is more accurate and in same ratio with the protected attribute
        normWidth = get_normalized_width(partition, i)
#       print("the selected QI is:", QID_NAMES[i])
#THIS IS HERE FOR WHEN I HAVE COLLECTED THE ORIGINAL ORDER SO I CAN COMPARE
        if QID_NAMES[i] in Protected_att:
#            print("in choose_dimension Protected_att:", QID_NAMES[i])
            ratio = altRatio(partition, QID_NAMES[i], goal, outcome)
            normWidth = 0.5*normWidth + 0.5*(1-ratio)
        else:
            normWidth = 0.5*normWidth

        if normWidth > max_width:
            max_width = normWidth
            max_dim = i
    if max_width > 1:
        print("Error: max_width > 1")
        pdb.set_trace()
    if max_dim == -1:
        print("cannot find the max dim")
        pdb.set_trace()
#    print(f"Selected dimension (quasi-identifier attribute): {ATT_NAMES[max_dim]}")
    print("the eventually selected QI is:", QID_NAMES[max_dim])
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dim
    return dict{key: str values, values: count}
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(partition, dim):
    """
    find the middle of the partition
    return splitVal
    """
    frequency = frequency_set(partition, dim)
    splitVal = ''
    # value_list = list(frequency)
    # value_list.sort(key=cmp_to_key(cmp_str))
    # total = sum(frequency.values())
    # middle = total / 2  

    value_list = frequency.keys()
    value_list.sort(cmp=cmp_str)
    total = sum(frequency.values())
    middle = total / 2

    if GL_L != 0:
        if middle < GL_L or len(value_list) <= 1:
            return ('', '', value_list[0], value_list[-1])
    elif GL_K != 0:
        if middle < GL_K or len(value_list) <= 1:
            return ('', '', value_list[0], value_list[-1])
    index = 0
    split_index = 0
    for i, t in enumerate(value_list):
        index += frequency[t]
        if index >= middle:
            splitVal = t
            split_index = i
            break
    else:
        print("Error: cannot find splitVal")
    try:
        nextVal = value_list[split_index + 1]
    except IndexError:
        nextVal = splitVal
    return (splitVal, nextVal, value_list[0], value_list[-1])


def split_numerical_value(numeric_value, splitVal):
    """
    split numeric value on splitVal
    return sub ranges
    """
    split_num = numeric_value.split(',')
    if len(split_num) <= 1:
        return split_num[0], split_num[0]
    else:
        low = split_num[0]
        high = split_num[1]
        # Fix 2,2 problem
        if low == splitVal:
            lvalue = low
        else:
            lvalue = low + ',' + splitVal
        if high == splitVal:
            rvalue = high
        else:
            rvalue = splitVal + ',' + high
        return lvalue, rvalue


def split_numerical(partition, dim, pwidth, pmiddle):
    """
    strict split numeric attribute by finding a median,
    lhs = [low, means], rhs = (mean, high]
    """
    sub_partitions = []
    # numeric attributes
    (splitVal, nextVal, low, high) = find_median(partition, dim)
    p_low = ATT_TREES[dim].dict[low]
    p_high = ATT_TREES[dim].dict[high]
    # update middle
    if low == high:
        pmiddle[dim] = low
    else:
        pmiddle[dim] = low + ',' + high
    pwidth[dim] = (p_low, p_high)
    if splitVal == '' or splitVal == nextVal:
        # update middle
        return []
    middle_pos = ATT_TREES[dim].dict[splitVal]
    lmiddle = pmiddle[:]
    rmiddle = pmiddle[:]
    lmiddle[dim], rmiddle[dim] = split_numerical_value(pmiddle[dim], splitVal)
    lhs = []
    rhs = []
    for temp in partition.member:
        pos = ATT_TREES[dim].dict[temp[dim]]
        if pos <= middle_pos:
            # lhs = [low, means]
            lhs.append(temp)
        else:
            # rhs = (mean, high]
            rhs.append(temp)
    lwidth = pwidth[:]
    rwidth = pwidth[:]
    lwidth[dim] = (pwidth[dim][0], middle_pos)
    rwidth[dim] = (ATT_TREES[dim].dict[nextVal], pwidth[dim][1])
    if GL_L != 0:
        if check_L_diversity(lhs) is False or check_L_diversity(rhs) is False:
            return []
    sub_partitions.append(Partition(lhs, lwidth, lmiddle))
    sub_partitions.append(Partition(rhs, rwidth, rmiddle))
    return sub_partitions


def split_categorical(partition, dim, pwidth, pmiddle):
    """
    split categorical attribute using generalization hierarchy
    """
    sub_partitions = []
    # categoric attributes
    splitVal = ATT_TREES[dim][partition.middle[dim]]
    sub_node = [t for t in splitVal.child]
    sub_groups = []
    for i in range(len(sub_node)):
        sub_groups.append([])
    if len(sub_groups) == 0:
        # split is not necessary
        return []
    for temp in partition.member:
        qid_value = temp[dim]
        for i, node in enumerate(sub_node):
            try:
                node.cover[qid_value]
                sub_groups[i].append(temp)
                break
            except KeyError:
                continue
        else:
            print("Generalization hierarchy error!: " + qid_value)
    flag = True
    for index, sub_group in enumerate(sub_groups):
        if len(sub_group) == 0:
            continue
        
        if GL_L != 0:
            if check_L_diversity(sub_group) is False:
                flag = False
                break
        elif GL_K != 0:
            if len(sub_group) < GL_K:
                flag = False
                break
    if flag:
        for i, sub_group in enumerate(sub_groups):
            if len(sub_group) == 0:
                continue
            wtemp = pwidth[:]
            mtemp = pmiddle[:]
            wtemp[dim] = len(sub_node[i])
            mtemp[dim] = sub_node[i].value
            sub_partitions.append(Partition(sub_group, wtemp, mtemp))
    return sub_partitions


def split_partition(partition, dim):
    """
    split partition and distribute records to different sub-partitions
    """
    pwidth = partition.width
    pmiddle = partition.middle
    if IS_CAT[dim] is False:
        return split_numerical(partition, dim, pwidth, pmiddle)
    else:
        return split_categorical(partition, dim, pwidth, pmiddle)


def anonymize(partition,QID_NAMES, Protected_att, goal, outcome):
    """
    Main procedure of Half_Partition.
    recursively partition groups until not allowable.
    """
#    print("Partition member records:")
#    for record in partition.member:
#        print(record)
    if check_splitable(partition) is False:
        RESULT.append(partition)
        return
    # Choose dim
    dim = choose_dimension(partition, QID_NAMES, Protected_att, goal, outcome)
    if dim == -1:
        print("Error: dim=-1")
        pdb.set_trace()
    sub_partitions = split_partition(partition, dim)
    if len(sub_partitions) == 0:
        partition.allow[dim] = 0
        anonymize(partition, QID_NAMES, Protected_att, goal, outcome)
    else:
        for sub_p in sub_partitions:
            anonymize(sub_p, QID_NAMES, Protected_att, goal, outcome)


def check_splitable(partition):
    """
    Check if the partition can be further splited while satisfying k-anonymity.
    """
    temp = sum(partition.allow)
    if temp == 0:
        return False
    return True


def init(att_trees, data, QI_num, SA_num, k=None, L=None):
    """
    reset all global variables
    """
    global GL_K, RESULT, QI_LEN, ATT_TREES, QI_RANGE, IS_CAT, SA_INDEX, GL_L
    ATT_TREES = att_trees
    for t in att_trees:
        if isinstance(t, NumRange):
            IS_CAT.append(False)
        else:
            IS_CAT.append(True)

    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num

    SA_INDEX = SA_num
    RESULT = []
    QI_RANGE = []
    if k is not None:
        GL_K = k
    else:
        GL_K = 0
    if L is not None:
        GL_L = L
    else:
        GL_L = 0

def check_L_diversity(partition):
    """check if partition satisfy l-diversity
    return True if satisfy, False if not.
    """
    sa_dict = {}
    if len(partition) < GL_L:
        return False
    if isinstance(partition, Partition):
        records_set = partition.member
    else:
        records_set = partition
    num_record = len(records_set)
    for record in records_set:
        sa_value = record[-1]
        try:
            sa_dict[sa_value] += 1
        except KeyError:
            sa_dict[sa_value] = 1
    if len(sa_dict.keys()) < GL_L:
        return False
    for sa in sa_dict.keys():
        # if any SA value appear more than |T|/l,
        # the partition does not satisfy l-diversity
        if sa_dict[sa] > 1.0 * num_record / GL_L:
            return False
    return True

def mondrian(att_trees, data, k, QI_num, SA_num,ATT_NAMES, QID_NAMES, Protected_att, goal, outcome):
    """
    basic Mondrian for k-anonymity.
    This fuction support both numeric values and categoric values.
    For numeric values, each iterator is a mean split.
    For categoric values, each iterator is a split on GH.
    The final result is returned in 2-dimensional list.
    """
    init(att_trees, data, QI_num, SA_num, k=k)
    result = []
    middle = []
    wtemp = []
    print("this is the print in modified_mondrian")
    #print(att_trees)

#   quasi_identifier_names = [data.columns[i] for i in range(QI_LEN)]
#    print(QI_LEN)
#    print(QI_num)
#    print(k)
#    print(IS_CAT)
#    print(SA_INDEX)
#    print(SA_num)
    print("this is att_names in modified_mondrian:", ATT_NAMES)
    print("this is QID_NAMES in modified mondrian:", QID_NAMES)
    print("this is portected in modified mondrian:", Protected_att)
    print("this is goal in modified mondrian:", goal)
    print("this is outcome in modified mondrian:",outcome)
    print("ITWORKSAGIAN")
#    print(data)

    for i in tqdm(range(QI_LEN)):
        if IS_CAT[i] is False:
            QI_RANGE.append(ATT_TREES[i].range)
            wtemp.append((0, len(ATT_TREES[i].sort_value) - 1))
            middle.append(ATT_TREES[i].value)
        else:
            QI_RANGE.append(len(ATT_TREES[i]['*']))
            wtemp.append(len(ATT_TREES[i]['*']))
            middle.append('*')
    whole_partition = Partition(data, wtemp, middle)
    start_time = time.time()
    anonymize(whole_partition,QID_NAMES, Protected_att, goal, outcome)
    rtime = float(time.time() - start_time)
    for partition in RESULT:
        temp = partition.middle
        for i in range(len(partition)):
            temp_for_SA = []
            for s in range(len(partition.member[i]) - len(SA_INDEX), len(partition.member[i])):
                temp_for_SA = temp_for_SA + [partition.member[i][s]]
            result.append(temp + temp_for_SA)
    return (result, rtime)

def mondrian_l_diversity(att_trees, data, L, QI_num, SA_num):
    """
    Mondrian for l-diversity.
    This fuction support both numeric values and categoric values.
    For numeric values, each iterator is a mean split.
    For categoric values, each iterator is a split on GH.
    The final result is returned in 2-dimensional list.
    """
    init(att_trees, data, QI_num, SA_num, L=L)
    middle = []
    result = []
    wtemp = []
    for i in range(QI_LEN):
        if IS_CAT[i] is False:
            QI_RANGE.append(ATT_TREES[i].range)
            wtemp.append((0, len(ATT_TREES[i].sort_value) - 1))
            middle.append(ATT_TREES[i].value)
        else:
            QI_RANGE.append(len(ATT_TREES[i]['*']))
            wtemp.append(len(ATT_TREES[i]['*']))
            middle.append('*')
    whole_partition = Partition(data, wtemp, middle)
    start_time = time.time()
    anonymize(whole_partition)
    rtime = float(time.time() - start_time)
 
    dp = 0.0
    for partition in RESULT:
        dp += len(partition) ** 2
        temp = partition.middle
        for i in range(len(partition)):
            temp_for_SA = []
            for s in range(len(partition.member[i]) - len(SA_INDEX), len(partition.member[i])):
                temp_for_SA = temp_for_SA + [partition.member[i][s]]
            result.append(temp + temp_for_SA)

    return (result, rtime)
