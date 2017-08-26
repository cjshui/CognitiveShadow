import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 16})

def testing(entree, frequence_table, sample, threshold):
    """
    Here the sample has three choices: 1, 2 and 3
    
    """
    # the fre_table contain the number of occurance in the table
    fre_table = frequence_table[entree,:]
    
    # then we model the parameter in posterori beta_distribution 
    alpha =  1 + fre_table[sample-1]
    beta = 1 + fre_table.sum() - fre_table[sample-1]
    
    # We compute the mean and variance of our posterori beta_distribution
    average = alpha *1.0/ (alpha + beta)*1.0
    variance = alpha*beta*1.0 / ((alpha+beta)**2*(alpha+beta+1))*1.0
                                
    # the average here is the expected reawrd that the we choose the sample ~(0,1) 
    # here if the upper bound UCB (setting as \mu + \sigma) is supper than threshold, we accept (return 1)
    # otherwise, we set up an alert (0)
    
    seuil  =  average + np.sqrt(variance)
    
    # seuil = average
    
    return np.int(seuil>=threshold)

def best_testing(entree, frequence_table, sample):
    """
    the best_testing we set a thresholding with the best expectation value as threshold
    computing the ucb
    :param entree:
    :param frequence_table:
    :param sample:
    :return:
    """
    # the fre_table contain the number of occurance in the table
    fre_table = frequence_table[entree, :]

    # then we model the parameter in posterori beta_distribution
    alpha = 1 + fre_table[sample - 1]
    beta = 2 + fre_table.sum() - fre_table[sample - 1]

    # We compute the mean and variance of our posterori beta_distribution
    average = alpha * 1.0 / (alpha + beta) * 1.0
    variance = alpha * beta * 1.0 / ((alpha + beta) ** 2 * (alpha + beta + 1)) * 1.0

    # the average here is the expected reawrd that the we choose the sample ~(0,1)
    # here if the upper bound UCB (setting as \mu + \sigma) is supper than threshold, we accept (return 1)
    # otherwise, we set up an alert (0)

    seuil = average + np.sqrt(variance)

    threshold = fre_table.max()/(fre_table.sum())

    return np.int(seuil >= threshold)





def reject_testing(entree, frequence_table, sample, threshold):

    # the fre_table contain the number of occurance in the table
    fre_table = frequence_table[entree, :]

    # then we model the parameter in posterori beta_distribution
    alpha = 1 + fre_table[sample - 1]
    beta = 1 + fre_table.sum() - fre_table[sample - 1]

    # We compute the mean and variance of our posterori beta_distribution
    average = alpha * 1.0 / (alpha + beta) * 1.0
    variance = alpha * beta * 1.0 / ((alpha + beta) ** 2 * (alpha + beta + 1)) * 1.0

    # the average here is the expected reawrd that the we choose the sample ~(0,1)
    # here if the upper bound UCB (setting as \mu + \sigma) is supper than threshold, we accept (return 1)
    # otherwise, we set up an alert (0)

    seuil = average + np.sqrt(variance)

    # seuil = average

    return 1- np.int(seuil <= threshold)




def non_para_testing(entree, frequence_table, sample):
    """
        Here the sample has three choices: 1, 2 and 3

    """
    # the fre_table contain the number of occurance in the table
    fre_table = frequence_table[entree, :]

    # then we model the parameter in posterori beta_distribution
    alpha = 1 + fre_table[sample - 1]
    beta = 1 + fre_table.sum() - fre_table[sample - 1]

    # We compute the mean and variance of our posterori beta_distribution
    average = alpha * 1.0 / (alpha + beta) * 1.0

    candi = np.array([0,1,2])
    candi = np.delete(candi,sample-1)
    other_average = np.zeros(2)

    for index in np.arange(2):
        alpha= 1 + fre_table[candi[index]]
        beta= 1 + fre_table.sum() - fre_table[candi[index]]
        other_average[index] = alpha * 1.0 / (alpha + beta) * 1.0

    return np.int(average >= np.max(other_average))




def checking(alert_signal, sample):
    
    """
    The alert_signal is 0 : means an alarm; 1: meaning normal
    There are four labels for checking the accuracy:
        -- 0   Alert shows normal, real case normal
        -- 1   Alert shows normal, real case NO--Wrong
        -- 2   Alert shows NO-normal, real case NO-normal
        -- 3   Alert shows NO-normal, real case Normal
    """
    
    if alert_signal == 1:

        if sample[0]== sample[1]:
            return 0
        else:
            return 1
    else:
        if sample[0]!=sample[1]:
            return 2
        else:
            return 3
        
def updating(entree, sample, frequency_table):
    
    """
    The new input will modify the frequency_table, probability_table
    
    Output: the frequency_table and probability_table
    
    """
    
    frequency_table[entree,sample-1]  = frequency_table[entree,sample-1]  + 1
                   
    return frequency_table            
    
def drawing(array):
    """
    The following function will draw two plots
    -- the correlation matrix in global level
    -- the points in ROC space
    the array is a 60 size list
    
    """    
    N_par = len(array)
    corr = np.zeros([2,2])
    ROC_point = np.zeros([N_par,2])

    for index in np.arange(N_par):
        
        result = array[index]
        
        corr[0,0] = corr[0,0] + np.sum(result==0)
        corr[0,1] = corr[0,1] + np.sum(result==3)
        corr[1,0] = corr[1,0] + np.sum(result==1)
        corr[1,1] = corr[1,1] + np.sum(result==2)
        
        TPR = np.sum(result==0) * 1.0/ (np.sum(result==0)+np.sum(result==3)) * 1.0
        FPR = np.sum(result==1) * 1.0/ (np.sum(result==1)+np.sum(result==2)) * 1.0
                    
        ROC_point[index,:]  = np.array([FPR,TPR])
    
    
    corr = corr.astype('float') / corr.sum(axis=1)[:, np.newaxis] 
    thresh = corr.max() / 2.
    text_ij = [['TP','FN'],['FP','TN']]
                 
                     
    plt.figure(0)
    plt.imshow(corr,interpolation='nearest',cmap=plt.cm.Blues)     
    for i, j in itertools.product(range(corr.shape[0]), range(corr.shape[1])):
        plt.text(j, i, np.around(corr[i,j],decimals=3),  \
                 verticalalignment="bottom",  \
                 color="white" if corr[i, j] > thresh else "black")
        plt.text(j, i, text_ij[i][j],  \
                 verticalalignment="top",  \
                 color="white" if corr[i, j] > thresh else "black")
        
        
    plt.colorbar(ticks=[0.05,0.1,0.5,0.9,0.95])
    x = np.array([0,1])
    y = x.copy()
    my_ticks = ['Normal','Abnormal']   
    plt.xticks(x, my_ticks)
    plt.yticks(y, my_ticks)
    plt.ylabel('True Condition')
    plt.xlabel('Predicted Condition')
    plt.tight_layout()
    # plt.show()
    # plt.savefig('cor_par_bad_2', bbox_inches='tight')
    plt.savefig('cor_par_bad_2.pdf', bbox_inches='tight',format='pdf', dpi=1000)
    
    # plt.figure(1)
    # plt.plot(ROC_point[:,0],ROC_point[:,1],'o')
    # plt.plot(np.arange(0,1.5,0.5),np.arange(0,1.5,0.5),'-')
    # plt.xlim([-0.01,1.01])
    # plt.ylim([-0.01,1.01])
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.show()
    # plt.savefig('ROC_par_bad_2',bbox_inches='tight')
    # plt.savefig('ROC_par_bad_2.pdf', bbox_inches='tight',format='pdf', dpi=1000)
    
    return None
        
        
def roc_curve_drawing(record):
    """
    The roc_curve_drawing will draw the roc curve with the different values

    :param record: list contains the participants's records
    :return:
    """
    Num_par = len(record)
    ROC_point = np.zeros([Num_par,21,2])
    for index_par in np.arange(Num_par):
        result = record[index_par]  # the result is a two dim array

        for elements in np.arange(21):
            TPR = np.sum(result[:,elements] == 0) * 1.0 / (np.sum(result[:,elements] == 0) + np.sum(result[:,elements] == 3)) * 1.0
            FPR = np.sum(result[:,elements] == 1) * 1.0 / (np.sum(result[:,elements] == 1) + np.sum(result[:,elements] == 2)) * 1.0


            ROC_point[index_par,elements,:] = np.array([FPR, TPR])


    color=iter(plt.cm.rainbow(np.linspace(0,1,Num_par)))
    # the next step we will draw the ROC curve with the results in ROC_point
    plt.figure(0)
    for index_par in np.arange(Num_par):
        c = next(color)
        plt.plot(ROC_point[index_par,:,0],ROC_point[index_par,:,1],'-',color = c,linewidth =2)


    plt.plot(np.arange(0, 1.5, 0.5), np.arange(0, 1.5, 0.5), '--', linewidth =2)
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.show()
    # plt.savefig('ROC_all_2',bbox_inches='tight')
    plt.savefig('ROC_all_2.pdf', bbox_inches='tight',format='pdf', dpi=1000)



    return None



def error_visu(result):
    """
    The function here will help us visulize the situations of detections and results
        during the streaming data
    :param result: Here we suppose that only one participant has involved, containing 0,1,2,3 four states
    :return:
    """
    data = result[0]
    S = np.zeros([len(data),2])
    S[:,1]  = data
    S[:,0]  = np.arange(len(data))

    TP = S[S[:, 1] == 0]
    FP = S[S[:, 1] == 1]
    FN = S[S[:, 1] == 3]
    TN = S[S[:, 1] == 2]

    # sns.set_style("dark")
    plt.figure(2)
    tp, = plt.plot(TP[:,0],TP[:,1],'.',color = 'blue',label = 'TP')
    fp, = plt.plot(FP[:, 0], FP[:, 1], '.', color='red', label='FP')
    fn, = plt.plot(FN[:, 0], FN[:, 1], 'v', color='blue', label='FN')
    tn, = plt.plot(TN[:, 0], TN[:, 1], 'v', color='red', label='TN')

    plt.legend(handles=[fn,tn,tp,fp])
    plt.ylim([-0.1,4])
    plt.xlim([-0.5,len(data)+0.5])
    plt.xlabel('Data Streaming')
    plt.yticks([])
    # plt.show()
    # plt.savefig('Situation_par_bad_2', bbox_inches='tight')
    plt.savefig('Situation_par_bad_2.pdf', bbox_inches='tight',format='pdf', dpi=1000)
