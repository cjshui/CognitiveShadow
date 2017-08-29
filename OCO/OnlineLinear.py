################################################
#  The approach will realize the online linear #
#  discriminative model without kernel         #
################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


class classifier:

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
                self.usrOriState.append(self.usrData[k][:,3:8])
                self.usrState.append(transform)
                self.usrDecision.append(self.usrData[k][:, 1])
                self.groundTruth.append(self.usrData[k][:, 2])

        self.predication = []

    def OnlineSvmFit(self, X, y, lam = 0.1, rate=0.1):
        """
        This part we will develop the linear svm part for online optimization
        :param X : the global input (with dim T*N), here the data comes in streaming way, X_1*, X_2*....
        :param Y : the corresponding prediction with {1,2...K}
        :param rate: learning rate during the algorithm
        :param lamdba: regularization coefficient
        :return: realTimePred returning the real time predication in the linear function

        """

        K = np.max(y)
        T = X.shape[0]
        dim = X.shape[1]  # feature dim
        W = 1/(np.sqrt(dim+1))*np.ones([dim+1, K])  # dim* (K+1) coefficient vectors
        realTimePred = np.zeros(T)

        for t in range(T):
            x_t = np.ones(dim+1)
            x_t[1:] = X[t,:]
            score = np.dot(W.T,x_t)
            realTimePred[t] = np.argmax(score) + 1

            W = (1-lam*rate)*W

            y_t = y[t]
            W[:,y_t-1] = W[:,y_t-1] + rate*x_t.T

            if realTimePred[t] != y_t:
                 W[:,int(realTimePred[t]-1)] = W[:,int(realTimePred[t]-1)] - rate*x_t.T
            else:
                 index = np.argsort(score)
                 W[:,int(index[-2])] = W[:,int(index[-2])] - rate*x_t.T

            for k in range(K):
                if np.linalg.norm(W[:,k]) >= np.sqrt(1/(lam)):
                    W[:,k] = W[:,k]* np.sqrt(1/(lam))/(np.linalg.norm(W[:,k]))


        return realTimePred

    def visuOnetime(self,yPred,yTrue):
        """

        :param yPred: predication result
        :param yTrue: ground truth
        :return:
        """
        plt.figure()
        N = len(yPred)
        f1 = plt.scatter(range(N),yPred,color = "r",label="predication")
        f2 = plt.scatter(range(N),yTrue,color = "b",label="GroundTruth")
        plt.legend()
        plt.savefig('par.pdf', bbox_inches='tight',format='pdf', dpi=1000)


        return None

def main():
    data = np.load('data_d_co.npy')
    cf = classifier(data,dataIndex=range(60))
    #XTrain = cf.data[:,3:8]
    #YTrain = cf.data[:,1]
    #YTrue  = cf.data[:,2]
    #predication = cf.OnlineSvmFit(XTrain, YTrain)
    predication = cf.OnlineSvmFit(cf.usrOriState[0],cf.usrDecision[0])
    #flag = predication!=cf.groundTruth[0]
    #print(flag)
    print(np.sum(predication==cf.groundTruth[0])/len(cf.groundTruth[0]))
    #print(np.sum(predication==YTrue)/len(YTrue))


if __name__ == "__main__":
    main()