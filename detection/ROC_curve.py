"""
Differenting from the ucb-detection, here we will present the ROC curve
with different threshold in UCB algorithm

"""
import numpy as np
import detection

d = np.load('data_d_co.npy')

# par_indice = np.array([106])

par_indice = np.unique(d[:,0])
results = []
problem_indice = np.array([5,33,41,47,58])
thres_interval = np.arange(0,1.05,0.05)

for indice, label in enumerate(par_indice):

    current_seg = d[d[:, 0] == label]


    # 2^5 = 32, 3 states with Hostiles, Uncertain, NON Hostiles
    # Here we initialise the frequency table and probability table
    # the probability_table will change if new elements come in the frequency table

    frequence_table = np.ones([32, 3])

    transform = current_seg[:, 3] * (2 ** 4) + current_seg[:, 4] * (2 ** 3) + \
                current_seg[:, 5] * (2 ** 2) \
                + current_seg[:, 6] * 2 + current_seg[:, 7]

    # We set up an alarm table, which includes two columns, for the
    # first columns, we save the alarm state value,
    # i.e: 1 (normal) No alarm, 0 alarm (not normal)
    # second columns, we evaluate the decision of the system is correct or not


    alarm = np.zeros(len(transform))
    # we set a record function here, for saving the cases in different threshold

    record = np.zeros([len(transform), len(thres_interval)])
    for index_threshold, threshold in enumerate(thres_interval):
        for index_trans, value_trans in enumerate(transform):
            alarm[index_trans] = detection.testing(transform[index_trans], frequence_table, current_seg[index_trans, 1],
                                                  threshold)

            # alarm[index_trans, 0] = detection.non_para_testing(transform[index_trans], frequence_table, current_seg[index_trans, 1])

            record[index_trans, index_threshold] = detection.checking(alarm[index_trans], current_seg[index_trans, 1:3])

            frequence_table = detection.updating(transform[index_trans], current_seg[index_trans, 1],
                                                                frequence_table)
    if indice not in problem_indice:
        results.append(record)

    # results.append(record)

detection.drawing(results)

