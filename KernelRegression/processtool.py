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

def preproAdult():

    file = pd.read_csv('/gel/usr/chshu1/Music/CognitiveShadow/data/adult.data',',')
    adult = np.array(file)
    row, _ = np.where(adult==' ?')
    missing_index = np.unique(row)
    # data cleaning
    clean_adult = np.delete(adult,missing_index,axis=0)

    NUMBER_SAMPLE = len(clean_adult[:,14])
    result = np.zeros(NUMBER_SAMPLE)

    result[clean_adult[:,14]==' >50K']= 2
    result[clean_adult[:,14]==' <=50K'] = 1

    numIndex = [0,2,4,10,11,12]
    # numercialFeature = clean_adult[:,numIndex]/np.max(clean_adult[:,numIndex],axis=0)
    numercialFeature = clean_adult[:, numIndex]
    cateFeature = [1,3,5,6,7,8,9]

    cate = [
        [' Private', ' Self-emp-not-inc', ' Self-emp-inc', ' Federal-gov', ' Local-gov', ' State-gov', ' Without-pay', ' Never-worked'],
        [' Bachelors',' Some-college', ' 11th', ' HS-grad', ' Prof-school',
        ' Assoc-acdm', ' Assoc-voc', ' 9th', ' 7th-8th', ' 12th', ' Masters', ' 1st-4th', ' 10th', ' Doctorate', ' 5th-6th', ' Preschool'],
        [' Married-civ-spouse', ' Divorced', ' Never-married', ' Separated', ' Widowed', ' Married-spouse-absent', ' Married-AF-spouse'],
        [' Tech-support', ' Craft-repair', ' Other-service', ' Sales', ' Exec-managerial',' Prof-specialty', ' Handlers-cleaners',
        ' Machine-op-inspct', ' Adm-clerical', ' Farming-fishing', ' Transport-moving', ' Priv-house-serv', ' Protective-serv',' Armed-Forces'],
        [' Wife', ' Own-child', ' Husband', ' Not-in-family', ' Other-relative', ' Unmarried'],
        [' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other', ' Black'],
        [' Female', ' Male']
    ]

    enc = OneHotEncoder(sparse=False)
    le = LabelEncoder()
    temp1 = np.zeros([NUMBER_SAMPLE,7])

    for i in range(7):

        index = cateFeature[i]
        data = clean_adult[:, index]
        le.fit(cate[i])
        temp1[:, i] = le.transform(data)

    enc.fit(temp1)
    categFeature = enc.transform(temp1)

    feature = np.concatenate((categFeature,numercialFeature),axis=1)

    return feature,result

def preprobank():
    """
    The reprocessing function is more complicated with the previous case,
    here we simply use the one-hot coding for processing the categorical variable,
    then computing the features
    """
    file = pd.read_csv('/gel/usr/chshu1/Music/CognitiveShadow/data/bank.csv',';')
    bank = np.array(file.values)
    NUMBER_SAMPLE = len(bank[:,16])

    result = np.zeros(NUMBER_SAMPLE)
    result[bank[:,16]=='yes']= 2
    result[bank[:,16]=='no'] = 1

    yesnoFeature = bank[:,[4,6,7]]
    yesnoConvert = np.zeros([NUMBER_SAMPLE,3])
    yesnoConvert[yesnoFeature=='yes']= 1
    yesnoConvert[yesnoFeature=='no'] = 0

    numIndex = [0,5,9,11,12,13,14]
    numercialFeature = bank[:,numIndex]

    # for feature 13, -1 means never contacted we set as maxvalue * 2

    numercialFeature[numercialFeature[:,5]==-1,5] = np.max(numercialFeature[:,5]) * 2

    enc = OneHotEncoder(sparse=False)
    le = LabelEncoder()
    categFeature = [1,2,3,8,10,15]
    cate_1 = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                        "blue-collar","self-employed","retired","technician","services"]
    cate_2 = ["married","divorced","single"]
    cate_3 = ["unknown","secondary","primary","tertiary"]
    cate_8 = [ "unknown","telephone","cellular"]
    cate_10 = ["jan", "feb", "mar", "apr","may","jun","jul","aug","sep","oct", "nov", "dec"]
    cate_15 = ["unknown","other","failure","success"]
    cate = []
    cate.append(cate_1)
    cate.append(cate_2)
    cate.append(cate_3)
    cate.append(cate_8)
    cate.append(cate_10)
    cate.append(cate_15)
    temp1 = np.zeros([NUMBER_SAMPLE,4])

    for i in range(4):
        index = categFeature[i]
        data = bank[:,index]
        le.fit(cate[i])
        temp1[:,i] = le.transform(data)


    enc.fit(temp1)
    cateFeature = enc.transform(temp1)
    # feature normalization
    # cateFeature[:12] = (1/12)* cateFeature[:12]
    # cateFeature[:,12:15]= cateFeature[:,12:15]/3
    # cateFeature[:,15:19]= cateFeature[:,15:19]/4
    # cateFeature[:,19:]= cateFeature[:,19:]/4

    feature = np.concatenate((yesnoConvert,cateFeature,numercialFeature),axis=1)

    return feature,result




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





