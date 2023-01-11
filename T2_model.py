from random import shuffle
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
import sklearn
import xlwt as xlwt
from skfeature.function.similarity_based import fisher_score, reliefF
from skfeature.function.statistical_based import t_score
from sklearn import svm, model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import LeaveOneOut, train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import keras
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier


def get_features_from_csv(pet5, pet6, pet16, pet17, pet11, pet21, t1, t2, clinical):
    feature_set = []
    label = []

    clinical_dataset = pd.read_csv(clinical)
    pp_id = clinical_dataset['name'].tolist()
    clinical_feature = dict(zip(pp_id, clinical_dataset.iloc[:, 1:-1].values))

    pet5_dataset = pd.read_csv(pet5)
    pp_t1c = pet5_dataset['name'].tolist()
    pet5_feature = dict(zip(pp_t1c, pet5_dataset.iloc[:, 38:-3].values))

    pet6_dataset = pd.read_csv(pet6)
    pp_t2f = pet6_dataset['name'].tolist()
    pet6_feature = dict(zip(pp_t2f, pet6_dataset.iloc[:, 38:-3].values))

    pet16_dataset = pd.read_csv(pet16)
    pp_t2f = pet16_dataset['name'].tolist()
    pet16_feature = dict(zip(pp_t2f, pet16_dataset.iloc[:, 38:-3].values))

    pet17_dataset = pd.read_csv(pet17)
    pp_t2f = pet17_dataset['name'].tolist()
    pet17_feature = dict(zip(pp_t2f, pet17_dataset.iloc[:, 38:-3].values))

    pet11_dataset = pd.read_csv(pet11)
    pp_t2f = pet11_dataset['name'].tolist()
    pet11_feature = dict(zip(pp_t2f, pet11_dataset.iloc[:, 38:-3].values))

    pet21_dataset = pd.read_csv(pet21)
    pp_t2f = pet21_dataset['name'].tolist()
    pet21_feature = dict(zip(pp_t2f, pet21_dataset.iloc[:, 38:-3].values))

    t1c_dataset = pd.read_csv(t1)
    pp_t1c = t1c_dataset['name'].tolist()
    t1c_feature = dict(zip(pp_t1c, t1c_dataset.iloc[:, 24:-3].values))

    t2f_dataset = pd.read_csv(t2)
    pp_t2f = t2f_dataset['name'].tolist()
    t2f_feature = dict(zip(pp_t2f, t2f_dataset.iloc[:, 24:-3].values))

    for key in clinical_feature.keys():
        _clinical = clinical_feature[key]
        try:
            _pet5 = pet5_feature[key]
            _pet6 = pet6_feature[key]
            _pet16 = pet16_feature[key]
            _pet17 = pet17_feature[key]
            _pet11 = pet11_feature[key]
            _pet21 = pet21_feature[key]
            _t1 = t1c_feature[key]
            _t2 = t2f_feature[key]

            temp = []
            """
            temp.extend(_pet5)
            temp.extend(_pet6)
            temp.extend(_pet16)
            temp.extend(_pet17)
            temp.extend(_pet11)
            temp.extend(_pet21)
            """
            # temp.extend(_t1)
            temp.extend(_t2)

            # temp.extend(_clinical[:-1])

            if int(_clinical[-1]) > 4.63:
                label.append(1)
            else:
                label.append(0)

            feature_set.append(temp)
            # print(temp)
        except:
            print('@@@@@@@@@@@@@@@@@@@@@@@@', key)
            continue

    return np.array(feature_set, dtype=np.float), np.array(label, dtype=np.int)


def get_performance(y_prob, y_true, th):
    pred = [1 if i > th else 0 for i in y_prob]
    print('Accuracy:', accuracy_score(y_true, pred))
    print('AUC:', roc_auc_score(y_true, y_prob))
    print('Sensitivity:', recall_score(y_true, pred))
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print('Specificity:', tn / (tn + fp))
    precision, recall, _thresholds = precision_recall_curve(y_true, y_prob)
    area = auc(recall, precision)
    print('PR AUC: ', area)


if __name__ == "__main__":
    pet5_file = '../csv/Harmonize_Institution/Institution-Non-EB/pet5_combat.csv'
    pet6_file = '../csv/Harmonize_Institution/Institution-Non-EB/pet6_combat.csv'
    pet16_file = '../csv/Harmonize_Institution/Institution-Non-EB/pet16_combat.csv'
    pet17_file = '../csv/Harmonize_Institution/Institution-Non-EB/pet17_combat.csv'
    pet11_file = '../csv/Harmonize_Institution/Institution-Non-EB/pet11_combat.csv'
    pet21_file = '../csv/Harmonize_Institution/Institution-Non-EB/pet21_combat.csv'
    t1 = '../csv/Harmonize_Institution/Institution-Non-EB/t1_combat.csv'
    t2 = '../csv/Harmonize_Institution/Institution-Non-EB/t2_combat.csv'

    clinical_file = '../csv/clinical_newest.csv'

    book = xlwt.Workbook()
    sheet = book.add_sheet('train_avg_auc')
    sheet_train = book.add_sheet('train_auc')
    sheet_validate = book.add_sheet('validate_auc')
    sheet_test = book.add_sheet('test_auc')
    r = 0
    c = 0
    seed = 0
    features, labels = get_features_from_csv(pet5_file, pet6_file, pet16_file, pet17_file, pet11_file, pet21_file, t1,
                                             t2,
                                             clinical_file)
    train, test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                    random_state=seed, stratify=labels)
    val, test, y_val, y_test = train_test_split(test, y_test, test_size=0.5,
                                                random_state=seed, stratify=y_test)

    mm = MinMaxScaler()
    mm.fit(train)
    train = mm.transform(train)
    val = mm.transform(val)
    test = mm.transform(test)

    num_fea = 40

    model = DecisionTreeClassifier(random_state=seed)
    sel = SelectKBest(t_score.t_score, k=num_fea)

    train = sel.fit_transform(train, y_train)
    val = sel.transform(val)
    test = sel.transform(test)

    model.fit(train, y_train)
    indexes = []
    for index, i in enumerate(sel.get_support()):
        if i:
            print(index, i)
            indexes.append(index)
    print(indexes)

    joblib.dump(sel, './model/T2_features.pkl')
    joblib.dump(model, './model/T2_model.pkl')

    Y_pred = model.predict_proba(val)
    fpr, tpr, thresholds = roc_curve(y_val.ravel(), Y_pred[:, -1].ravel())

    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = 0.5

    get_performance(Y_pred[:, -1], y_val, threshold)
    print(list(Y_pred[:, -1]))
    Y_pred = model.predict_proba(test)
    get_performance(Y_pred[:, -1], y_test, threshold)
    print(list(Y_pred[:, -1]))
