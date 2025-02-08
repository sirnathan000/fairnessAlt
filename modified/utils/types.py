# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum


class Dataset(Enum):
    CMC = 'cmc'
    MGM = 'mgm'
    ADULT = 'adult'
    ADULT2 = 'adult2'
    CAHOUSING = 'cahousing'
    INFORMS = 'informs'
    ITALIA = 'italia'
    FOLKSTABLE = 'folkstable'
    STUDENT = 'student'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class AnonMethod(Enum):

    #Optimal Lattice Anonymization
    OLA = 'ola'

    # Classic Mondrian (no hierchies)
    CLASSIC_MONDRIAN = 'classic_mondrian'

    # Basic Mondrian
    BASIC_MONDRIAN = 'mondrian'

    # modified Mondrian
    MODIFIED_MONDRIAN = 'modified_mondrian'

    # Alternative Modified Mondrian
    ALT_MODIFIED_MONDRIAN = 'alt_modified_mondrian'

    # Mondrian L-diversity
    MONDRIAN_LDIV = "mondrian_ldiv"
    
    # Top-Down Greedy
    TOPDOWN = 'topdown'

    # Cluster-based
    CLUSTER = 'cluster'

    # Datafly
    DATAFLY = 'datafly'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value

class ClassifierModel(Enum):
    SVM = 'svm'
    RF = 'rf'
    KNN = 'knn'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value
