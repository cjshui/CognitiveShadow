################################################################################
#  This file contains some functions in processing the data                    #
################################################################################

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder



def preproIRIS():
    """

    data description: the iris data has 149 samples, three classes: Iris-virginica,Iris-versicolor,Iris-setosa
    iris data-set 149 * 5

    :return:
    """
    file = pd.read_csv('/gel/usr/chshu1/Music/CognitiveShadow/data/iris.data',',')

    iris = np.array(file.values)


    np.random.shuffle(iris)

    label_str = iris[:,4]
    label = np.zeros(np.shape(label_str))
    feature = iris[:,:4]
    # normalization
    # feature = feature/np.max(feature,axis=0)
    label[label_str=='Iris-setosa']= 1
    label[label_str=='Iris-versicolor']= 2
    label[label_str=='Iris-virginica']= 3
    return feature, label


def preprohardware():
    """
    The pre processing procedure for hardware testing
    :return:
    """
    file = pd.read_csv('/gel/usr/chshu1/Music/CognitiveShadow/data/machine.data',',')
    machine = np.array(file.values)

    result = machine[:,8]
    observation = machine[:,2:8]
    # observation = observation/np.max(observation,axis=0)

    return observation,result


def preprowine():

    file = pd.read_csv('/gel/usr/chshu1/Music/CognitiveShadow/data/wine.data', ',')
    wine = np.array(file.values)
    np.random.shuffle(wine)
    result = wine[:, 0]
    observation = wine[:, 1:]

    return observation, result





def blurringDis(label,rate):
    """
    The blurring function which adds the random perturbed labels
    :param label: the ground truth label
    :param rate: the changing probability for a certain label
    :return: the perturbing labels
    """

    N = len(label)
    flag = np.random.binomial(1,rate,N)
    errorNum  = np.sum(flag==1)
    labelBlur = np.copy(label)
    labelBlur[flag==1] = np.random.randint(1,high=np.max(label),size=errorNum)

    return labelBlur


def blurringCon(value,rate,degree):
    """

    :param value: the continuous values
    :param rate: blurring rate
    :param degree: the noise level e.g x + degree *randn(..)
    :return: the blurring value

    """

    ## In the regression model, the noise means the adversial values
    flag = np.random.choice([0, 1], size=len(value), p=[rate, 1-rate])
    new_label = value.copy()
    new_label[flag == 0] += (2*np.random.binomial(1, 0.5)-1) * degree * (1+np.random.randn())

    return new_label


# observation, trueLabel = preprohardware()
# noiseLabel = blurringCon(trueLabel,0.15,50)
#
# print(trueLabel)
# print(noiseLabel)





