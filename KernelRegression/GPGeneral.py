################################################################################
#  The Gaussian kernel regression approach is a typical non parameter way      #
#  for estimating the mean and variance                                        #
#  In this approach, we tried to directly model three actions as values        #
#  But we should make two notes here, the scalable input (different scales     #
#  of our model                                                                #
################################################################################


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn import linear_model
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class GPregression:

    def __init__(self,kernelParameter):
        """

        :param kernelParameter: here we only set the kernel type:

        """

        self.kernel = kernelParameter


    def OnlineGP(self,X,y,shift=1):

        """

        :param X: input with n*d (n: # of observations, d: observation dim
        :param y: perturbed ground truth
        :param shift: beginning from the first "shift" observations (just facilitating in programming)
        :return:

        """
        T = X.shape[0]
        alert = np.zeros(T - shift)

        for t in np.arange(shift, T):
            x_t = X[:t, :]
            y_t = y[:t]

            reg = GaussianProcessRegressor(kernel=self.kernel)
            reg.fit(x_t, y_t)
            average, std = reg.predict(X[t, :], return_std=True)
            alert[t - shift] = (np.abs(y[t] - average) <= 0.4)


        return alert

    def OnlineGPC(self,X,y,shift=1):
        """

        :param X: global input
        :param y: ground truth
        :return: alert=1 OK , alert=0 senting a alert

        """
        T = X.shape[0]
        alert = np.zeros(T-shift)
        clf = GaussianProcessClassifier(kernel=self.kernel)

        for t in np.arange(shift,T):
            x_t = X[:t,:]
            y_t = y[:t]
            clf.fit(x_t,y_t)
            score = clf.predict_proba(X[t,:])
            # print(score)
            if np.max(y)>=3:
                alert[t - shift] = (score[0][int(y[t] - 1)] >= 1 / np.max(y))
            else:
                alert[t - shift] = (score[0][int(y[t] - 1)] >= 0.5)

        return alert

    def checking(self,alert_signal, usrDecision, groundTruth, shift=1):

        """
        The alert_signal is 0 : means an alarm; 1: meaning normal
        There are four labels for checking the accuracy:
            -- 0   Alert shows normal, real case normal
            -- 1   Alert shows normal, real case NO--Wrong
            -- 2   Alert shows NO-normal, real case NO-normal
            -- 3   Alert shows NO-normal, real case Normal
        Attention: the size of alert_signal will be T-1, and usrDecision and groundTruth will be T
        """

        result = np.zeros(len(alert_signal))

        for i in range(len(alert_signal)):
            if alert_signal[i] == 1:


                if usrDecision[i+shift] == groundTruth[i+shift]:

                    result[i] = 0
                else:
                    result[i] = 1
            else:


                if usrDecision[i+shift] != groundTruth[i+shift]:
                    result[i] = 2
                else:
                    result[i] = 3

        return result

    def drawing(self,array,name):
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
        plt.savefig(name, bbox_inches='tight', format='pdf', dpi=1000)

        return None

    def error_visu(self,result,name):
        """
        The function here will help us visulize the situations of detections and results
            during the streaming data
        :param result: Here we suppose that only one participant has involved, containing 0,1,2,3 four states
        :return:

        """
        data = result
        S = np.zeros([len(data), 2])
        S[:, 1] = data
        S[:, 0] = np.arange(len(data))

        TP = S[S[:, 1] == 0]
        FP = S[S[:, 1] == 1]
        FN = S[S[:, 1] == 3]
        TN = S[S[:, 1] == 2]

        plt.figure(2)
        tp, = plt.plot(TP[:, 0], TP[:, 1], '.', color='blue', label='TP')
        fp, = plt.plot(FP[:, 0], FP[:, 1], '.', color='red', label='FP')
        fn, = plt.plot(FN[:, 0], FN[:, 1], 'v', color='blue', label='FN')
        tn, = plt.plot(TN[:, 0], TN[:, 1], 'v', color='red', label='TN')

        plt.legend(handles=[fn, tn, tp, fp],loc=1)
        plt.ylim([-0.1, 4])
        plt.xlim([-0.5, len(data) + 0.5])
        plt.xlabel('Data Streaming')
        plt.yticks([])
        plt.savefig(name, bbox_inches='tight', format='pdf', dpi=1000)


class LogisticRegression:

    def __init__(self,multiclass=True):
        """

        :param multiclass: indicator with multiclassification, softmax function

        """
        self.mul = multiclass

    def logRe(self,X,y,shift=1):
        """

        :param X: input with n*d (n: # of observations, d: observation dim
        :param y: perturbed ground truth
        :param shift: beginning from the first "shift" observations (just facilitating in programming)
        :return:
        """

        if np.max(y) >=3 :
            clf = linear_model.LogisticRegression(multi_class='multinomial',solver='sag')
        else:
            clf = linear_model.LogisticRegression()

        T = X.shape[0]
        alert = np.zeros(T - shift)

        for t in np.arange(shift, T):
            x_t = X[:t, :]
            y_t = y[:t]

            clf.fit(x_t,y_t)
            score = clf.predict_proba(X[t,:])

            if np.max(y)>=3:
                alert[t - shift] = (score[0][int(y[t] - 1)] >= 1 / np.max(y))
            else:
                alert[t - shift] = (score[0][int(y[t] - 1)] >= 0.5)


        return alert

    def checking(self,alert_signal, usrDecision, groundTruth, shift=1):

        """
        The alert_signal is 0 : means an alarm; 1: meaning normal
        There are four labels for checking the accuracy:
            -- 0   Alert shows normal, real case normal
            -- 1   Alert shows normal, real case NO--Wrong
            -- 2   Alert shows NO-normal, real case NO-normal
            -- 3   Alert shows NO-normal, real case Normal
        Attention: the size of alert_signal will be T-1, and usrDecision and groundTruth will be T
        """

        result = np.zeros(len(alert_signal))

        for i in range(len(alert_signal)):
            if alert_signal[i] == 1:


                if usrDecision[i+shift] == groundTruth[i+shift]:

                    result[i] = 0
                else:
                    result[i] = 1
            else:


                if usrDecision[i+shift] != groundTruth[i+shift]:
                    result[i] = 2
                else:
                    result[i] = 3

        return result

    def drawing(self,array,name):
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
        plt.savefig(name, bbox_inches='tight', format='pdf', dpi=1000)

        return None

    def error_visu(self,result,name):
        """
        The function here will help us visulize the situations of detections and results
            during the streaming data
        :param result: Here we suppose that only one participant has involved, containing 0,1,2,3 four states
        :return:

        """
        data = result
        S = np.zeros([len(data), 2])
        S[:, 1] = data
        S[:, 0] = np.arange(len(data))

        TP = S[S[:, 1] == 0]
        FP = S[S[:, 1] == 1]
        FN = S[S[:, 1] == 3]
        TN = S[S[:, 1] == 2]

        plt.figure(2)
        tp, = plt.plot(TP[:, 0], TP[:, 1], '.', color='blue', label='TP')
        fp, = plt.plot(FP[:, 0], FP[:, 1], '.', color='red', label='FP')
        fn, = plt.plot(FN[:, 0], FN[:, 1], 'v', color='blue', label='FN')
        tn, = plt.plot(TN[:, 0], TN[:, 1], 'v', color='red', label='TN')

        plt.legend(handles=[fn, tn, tp, fp],loc=1)
        plt.ylim([-0.1, 4])
        plt.xlim([-0.5, len(data) + 0.5])
        plt.xlabel('Data Streaming')
        plt.yticks([])
        plt.savefig(name, bbox_inches='tight', format='pdf', dpi=1000)





















