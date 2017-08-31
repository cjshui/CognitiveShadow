################################################################################
#  This file contains some functions in processing the data                    #
################################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


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

def blurringCon(value,rate):
