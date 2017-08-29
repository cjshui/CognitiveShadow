################################################################################
#  The Gaussian kernel regression approach is a typical non parameter way      #
#  for estimating the mean and variance                                        #
#  In this approach, we tried to directly model three actions as values        #
################################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GPregrssor:

    def __init__(self,dataFile,dataIndex):
        """
                :param dataFile: file want to loaded, by defaut, as the CG model in thales
                :param dataIndex: the dataIdx is the data we want to treat, example: dataIdx = range(60), dataIdx = [5]

                """
        self.data = dataFile
        self.problemIndx = np.array([5, 33, 41, 47, 58])
        parInx = np.unique(self.data[:, 0])
        self.usrData = []
        for label in parInx:
            self.usrData.append(self.data[self.data[:, 0] == label])
        self.usrOriState = []
        self.usrState = []
        self.usrDecision = []
        self.groundTruth = []

        for k in range(len(self.usrData)):
            if k in dataIndex:
                # only extract the interest data measurement
                transform = self.usrData[k][:, 3] * (2 ** 4) + self.usrData[k][:, 4] * (2 ** 3) + \
                            self.usrData[k][:, 5] * (2 ** 2) \
                            + self.usrData[k][:, 6] * 2 + self.usrData[k][:, 7]
                self.usrOriState.append(self.usrData[k][:, 3:8])
                self.usrState.append(transform)
                self.usrDecision.append(self.usrData[k][:, 1])
                self.groundTruth.append(self.usrData[k][:, 2])



    def OnlineGP(self,X,y):
        """
        Online Gaussian Process regression

        :param X: global input
        :param y: predication
        :return prediction: the real time prediction and corresponding variance

        """

        T = X.shape[0]
        parameter = np.zeros([T-1,2])
        alert = np.zeros(T-1)

        for t in np.arange(1,T):
            x_t = X[:t,:]
            y_t = y[:t]

            reg = GaussianProcessRegressor()
            reg.fit(x_t,y_t)
            average, std = reg.predict(X[t,:],return_std=True)
            parameter[t-1,:] = [average,std]
            alert[t-1] = (np.abs(y[t]-average)<=3*std)

        return parameter,alert


    def checking(self,alert_signal, sample):

        """
        The alert_signal is 0 : means an alarm; 1: meaning normal
        There are four labels for checking the accuracy:
            -- 0   Alert shows normal, real case normal
            -- 1   Alert shows normal, real case NO--Wrong
            -- 2   Alert shows NO-normal, real case NO-normal
            -- 3   Alert shows NO-normal, real case Normal
        """

        if alert_signal == 1:

            if sample[0] == sample[1]:
                return 0
            else:
                return 1
        else:
            if sample[0] != sample[1]:
                return 2
            else:
                return 3

    def drawing(self,array):
        """
        The following function will draw two plots
        -- the correlation matrix in global level
        the array is a 60 size list
        """
        N_par = len(array)
        corr = np.zeros([2, 2])
        ROC_point = np.zeros([N_par, 2])

        for index in np.arange(N_par):
            result = array[index]

            corr[0, 0] = corr[0, 0] + np.sum(result == 0)
            corr[0, 1] = corr[0, 1] + np.sum(result == 3)
            corr[1, 0] = corr[1, 0] + np.sum(result == 1)
            corr[1, 1] = corr[1, 1] + np.sum(result == 2)

            TPR = np.sum(result == 0) * 1.0 / (np.sum(result == 0) + np.sum(result == 3)) * 1.0
            FPR = np.sum(result == 1) * 1.0 / (np.sum(result == 1) + np.sum(result == 2)) * 1.0

            ROC_point[index, :] = np.array([FPR, TPR])

        corr = corr.astype('float') / corr.sum(axis=1)[:, np.newaxis]
        thresh = corr.max() / 2.
        text_ij = [['TP', 'FN'], ['FP', 'TN']]

        plt.figure(0)
        plt.imshow(corr, interpolation='nearest', cmap=plt.cm.Blues)
        for i, j in itertools.product(range(corr.shape[0]), range(corr.shape[1])):
            plt.text(j, i, np.around(corr[i, j], decimals=3), \
                     verticalalignment="bottom", \
                     color="white" if corr[i, j] > thresh else "black")
            plt.text(j, i, text_ij[i][j], \
                     verticalalignment="top", \
                     color="white" if corr[i, j] > thresh else "black")

        plt.colorbar(ticks=[0.05, 0.1, 0.5, 0.9, 0.95])
        x = np.array([0, 1])
        y = x.copy()
        my_ticks = ['Normal', 'Abnormal']
        plt.xticks(x, my_ticks)
        plt.yticks(y, my_ticks)
        plt.ylabel('True Condition')
        plt.xlabel('Predicted Condition')
        plt.tight_layout()
        plt.savefig('cor_par_bad_2.pdf', bbox_inches='tight', format='pdf', dpi=1000)

        return None







def main():
    print('Loading Data')
    data = np.load('/gel/usr/chshu1/Music/CognitiveShadow/data/data_d_co.npy')

    print('Testing')
    cf = GPregrssor(data,dataIndex=range(60))
    #XTrain = cf.data[:,3:8]
    #YTrain = cf.data[:,1]
    #YTrue  = cf.data[:,2]
    #predication = cf.OnlineSvmFit(XTrain, YTrain)
    paramter,alert = cf.OnlineGP(cf.usrOriState[0],cf.usrDecision[0])

    print(paramter[:,0])
    print(alert)
    #print(np.sum(predication==cf.groundTruth[0])/len(cf.groundTruth[0]))
    #print(np.sum(predication==YTrue)/len(YTrue))


if __name__ == "__main__":
    main()