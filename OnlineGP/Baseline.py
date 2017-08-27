###########################################
#  The baseline will directly give a      #
#  Offline gaussian approach              #
##########################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score

d = np.load('data_d_co.npy')

par_indice = np.unique(d[:,0])
accu = np.zeros(par_indice.shape)
human = np.zeros(par_indice.shape)
cross_score =  np.zeros(par_indice.shape)

for index, label in enumerate(par_indice):
    current_seg = d[d[:,0]==label]
    clf  = GaussianProcessClassifier()
    clf.fit(current_seg[:,3:8],current_seg[:,1])
    # validation_scores = cross_val_score(clf,current_seg[:,3:8],current_seg[:,1],cv=10)
    accu[index] = clf.score(current_seg[:,3:8],current_seg[:,2])
    human[index] = np.sum(current_seg[:,8]==1)/current_seg.shape[0]
    # cross_score[index] = validation_scores.mean()



# print('The improvement of algorithm is ', np.sum(accu>=human)/60)
# print('The cross validation score is ', cross_score.mean(), 'the std of CV is ', cross_score.std())
# print('The average of algorithm is ', accu.mean(), 'the std of algorithm is ', accu.std())


svm_res = plt.scatter(np.arange(1,accu.shape[0]+1),accu,color='r',label='GP')
human_res= plt.scatter(np.arange(1,human.shape[0]+1),human,label='Participant')
plt.fill_between(np.arange(0,human.shape[0]+2), \
                 human.mean()-human.std(),human.mean()+human.std(),alpha = 0.2)
plt.xlabel('No of Participant')
plt.ylabel('Decision accuracy')
plt.title('Comparison')
plt.legend(handles=[svm_res,human_res])
plt.show()