#########################################################
#                        Demo                           #
#########################################################

from __future__ import print_function
import numpy as np
import processtool as pl
import GPGeneral as GPG
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern

### hardware example:

# observation, trueLabel = pl.preproIRIS()
# noiseLabel = pl.blurringDis(trueLabel,0.15)

# print(trueLabel)
# print(noiseLabel)


### Wine example
# observation, trueLabel = pl.preprowine()
# noiseLabel = pl.blurringDis(trueLabel,0.15)

### banke example
observation, trueLabel = pl.preprobank()
noiseLabel = pl.blurringDis(trueLabel,0.15)





### adult example
# observation, trueLabel = pl.preproAdult()
# noiseLabel = pl.blurringDis(trueLabel,0.15)


## GPC approach
# kernel =  Matern()
# cf = GPG.GPregression(kernelParameter=kernel)
# shift = 40
# alert = cf.OnlineGPC(observation[:500,:],noiseLabel[:500],shift=shift)
# evaluation = cf.checking(alert,noiseLabel[:500],trueLabel[:500],shift=shift)
# name1 = 'bankGPC.pdf'
# name2 = 'bankProGPC.pdf'
# cf.drawing(evaluation,name1)
# cf.error_visu(evaluation,name2)


## testing in logistic softmax regression
shift = 40
cf = GPG.LogisticRegression()

alert = cf.logRe(observation[:500,:],noiseLabel[:500],shift=shift)
evaluation = cf.checking(alert,noiseLabel[:500],trueLabel[:500],shift=shift)
name1 = 'bankLog.pdf'
name2 = 'bankProLog.pdf'
cf.drawing(evaluation,name1)
cf.error_visu(evaluation,name2)



