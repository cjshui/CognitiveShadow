#########################################################
#                        Demo                           #
#########################################################

from __future__ import print_function
import numpy as np
import processtool as pl
import GPGeneral as GPG
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern

### hardware example:

observation, trueLabel = pl.preproIRIS()
noiseLabel = pl.blurringDis(trueLabel,0.1)

kernel =  Matern()
cf = GPG.GPregression(kernelParameter=kernel)

shift = 5
alert = cf.OnlineGP(observation,noiseLabel,shift=shift)
print(alert)
evaluation = cf.checking(alert,noiseLabel,trueLabel,shift=shift)
name1 = 'test1GP.pdf'
name2 = 'Streamtest1GP.pdf'
cf.drawing(evaluation,name1)
cf.error_visu(evaluation,name2)



