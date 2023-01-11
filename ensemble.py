import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
from Code.T1 import get_performance

"""

"""

t1c_val = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

t1c_test = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

t2_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

t2_test = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

clinical_val = [0.4667023809523808, 0.0, 0.8649761904761903, 0.0, 0.5068333333333332, 0.5430238095238097, 0.73875,
                0.8636666666666667, 0.07166666666666666, 0.5430238095238097, 0.39363492063492034, 0.0, 0.02, 0.01, 0.0,
                0.39363492063492034, 0.03, 0.0, 0.1625, 0.687809523809524]

clinical_test = [0.0, 0.9455, 0.11, 0.765, 0.40688095238095245, 0.26175, 0.0, 0.9455, 0.3559999999999999,
                 0.5153095238095238, 0.4667023809523808, 0.40333333333333327, 0.23466666666666666, 0.9455,
                 0.007857142857142858, 0.07, 0.0, 0.3829574314574314, 0.17854761904761907, 0.07]

# clinical huozhe top2也行

pet_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
pet_test = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]

y_val = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
y_test = [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]

# a = 0.50
# b = 0.1
# c = 0.2
# d = 1 - 2b - c

a = 0.38
b = 0.0
c = 0.31
d = 1 - a - b - c

val = a * np.array(clinical_val) + b * np.array(t1c_val) + c * np.array(t2_val) + d * np.array(pet_val)
test = a * np.array(clinical_test) + b * np.array(t1c_test) + c * np.array(t2_test) + d * np.array(pet_test)

external_test = [0] * 11
external_test.extend([1] * 11)

external_t1 = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
external_t2 = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
external_pet = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
external_clinical = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0]

external_pred = a * np.array(external_clinical) + b * np.array(external_t1) + c * np.array(external_t2) + d * np.array(
    external_pet)

fpr, tpr, thresholds = roc_curve(y_val, val.ravel())
maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))

threshold = thresholds[maxindex]
print(threshold)
# threshold = 0.5
get_performance(val, y_val, threshold)
get_performance(test, y_test, threshold)

get_performance(external_pred, external_test, threshold)

fprt1, tprt1, _ = roc_curve(y_test, t1c_test)
rocauct1 = auc(fprt1, tprt1)
fprt2, tprt2, _ = roc_curve(y_test, t2_test)
rocauct2 = auc(fprt2, tprt2)
fprpet, tprpet, _ = roc_curve(y_test, pet_test)
rocaucpet = auc(fprpet, tprpet)
fprclinical, tprclinicl, _ = roc_curve(y_test, clinical_test)
rocaucclinical = auc(fprclinical, tprclinicl)
fprensemble, tprensemble, _ = roc_curve(y_test, test)
rocaucensemble = auc(fprensemble, tprensemble)

plt.figure()
lw = 2
plt.plot(fprt1, tprt1, color='darkorange',
         lw=lw, label='T1WI (area = %0.2f)' % rocauct1)
plt.plot(fprt2, tprt2, color='green',
         lw=lw, label='T2WI (area = %0.2f)' % rocauct2)
plt.plot(fprpet, tprpet, color='pink',
         lw=lw, label='DAT-SPECT (area = %0.2f)' % rocaucpet)
plt.plot(fprclinical, tprclinicl, color='blue',
         lw=lw, label='Clinical (area = %0.2f)' % rocaucclinical)
plt.plot(fprensemble, tprensemble, color='red',
         lw=lw, label='Ensemble (area = %0.2f)' % rocaucensemble)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

prt1, ret1, _ = precision_recall_curve(y_test, t1c_test)
prauct1 = auc(ret1, prt1)
prt2, ret2, _ = precision_recall_curve(y_test, t2_test)
prauct2 = auc(ret2, prt2)
prpet, repet, _ = precision_recall_curve(y_test, pet_test)
praucpet = auc(repet, prpet)
prclinical, reclinicl, _ = precision_recall_curve(y_test, clinical_test)
praucclinical = auc(reclinicl, prclinical)
prensemble, reensemble, _ = precision_recall_curve(y_test, test)
praucensemble = auc(reensemble, prensemble)

plt.figure()
lw = 2
plt.plot(ret1, prt1, color='darkorange',
         lw=lw, label='T1WI (area = %0.2f)' % prauct1)
plt.plot(ret2, prt2, color='green',
         lw=lw, label='T2WI (area = %0.2f)' % prauct2)
plt.plot(repet, prpet, color='pink',
         lw=lw, label='DAT-SPECT (area = %0.2f)' % praucpet)
plt.plot(reclinicl, prclinical, color='blue',
         lw=lw, label='Clinical (area = %0.2f)' % praucclinical)
plt.plot(prensemble, reensemble, color='red',
         lw=lw, label='Ensemble (area = %0.2f)' % praucensemble)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower left")
plt.show()
