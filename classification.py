import numpy as np
import pandas  as pd
import time
import gc
from scipy.sparse import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import  accuracy_score
from cela_feature_extraction import load 
from cola_feature_extraction import load as col_load
from helpers import get_datatype_stats


def classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key):
    print('training classifier...')
    if classifier=='RF':
        clf= RandomForestClassifier(random_state=42, max_depth=20, n_estimators=50)
        clf, duration_train=clf_fit(clf, X_train, y_train) 
        y_pred, duration_pred=clf_predict(clf, X_test)
        y_prob=clf.predict_proba(X_test)
    if classifier == 'LinSVM':
        clf = SVC(kernel='linear',random_state=42, probability=True)
        clf, duration_train=clf_fit(clf, X_train, y_train)
        y_pred, duration_pred=clf_predict(clf, X_test)
        y_prob=clf.predict_proba(X_test)
    if classifier == 'KNN':
        clf = KNeighborsClassifier()
        clf, duration_train=clf_fit(clf, X_train, y_train)
        y_pred, duration_pred=clf_predict(clf, X_test)
        y_prob=clf.predict_proba(X_test)
    if classifier == 'GNB':
        X_train=X_train.todense()
        X_test=X_test.todense()
        clf = GaussianNB()
        clf, duration_train=clf_fit(clf, X_train, y_train)
        y_pred, duration_pred=clf_predict(clf, X_test)
        y_prob=clf.predict_proba(X_test)
    if classifier=='LR':
        clf = LogisticRegression(random_state=42)
        clf, duration_train=clf_fit(clf, X_train, y_train)
        y_pred, duration_pred=clf_predict(clf, X_test)
        y_prob=clf.predict_proba(X_test)
    print('training done!')
    
    #ensure that only appearing lables are contained in labels
    labels_help=np.concatenate((y_test,y_pred))
    labels_help=np.unique(labels_help)
    labels_help=labels_help.tolist()
    labels=[dict_labels[i] for i in labels_help]
    
    report=classification_report(y_test, y_pred, target_names=labels, digits=3, zero_division=1) 
    cm=confusion_matrix(y_test, y_pred)
    df_cm=pd.DataFrame(cm, columns=labels, index=labels)
    precision,recall,f1score,support=score(y_test,y_pred,average='weighted', zero_division=1)
    accuracy=accuracy_score(y_test,y_pred)
    #mrr=mrr_score(y_test, y_prob)

    ser_scores_params=pd.Series(dtype=object)
    ser_scores_params['Database']=key
    ser_scores_params['# of properties']=len(dict_labels)
    ser_scores_params['Classifier']=classifier
    ser_scores_params['Accuracy']=accuracy
    ser_scores_params['Precision']=precision
    ser_scores_params['Recall']=recall
    ser_scores_params['F1-score']=f1score
    #ser_scores_params['MRR']=mrr
    ser_scores_params['Duration training']=duration_train
    ser_scores_params['Duration prediction']=duration_pred
    print()
    print(ser_scores_params)
    print()
    return ser_scores_params, df_cm, report, clf
    
def clf_fit(clf, X_train, y_train):
    tic = time.perf_counter()
    clf.fit(X_train, y_train)
    toc = time.perf_counter()
    duration=toc-tic
    return clf, duration

def clf_predict(clf, X_test):
    tic = time.perf_counter()
    y_pred=clf.predict(X_test)
    toc = time.perf_counter()
    duration=toc-tic
    return y_pred, duration  

def make_file(arr_scores_params, keys, classifiers, filename):
    file_name=filename
    startcol=0
    startrow=0
    with pd.ExcelWriter(file_name) as writer:
        for key in range(len(keys)):
            for classifier in range(len(classifiers)):
                if keys==[0]:
                    arr_scores_params[classifier].to_excel(writer,startrow=startrow, startcol=startcol, header=['{} {}'.format('train lld, test uniprot', classifiers[classifier])])
                    startrow+=arr_scores_params[classifier].size +2
                else:
                    arr_scores_params[classifier, key].to_excel(writer,startrow=startrow, startcol=startcol, header=['{} {}'.format(keys[key], classifiers[classifier])])
                    startrow+=arr_scores_params[classifier, key].size +2
            startcol+=3
            startrow=0
    
def main(samples, keys,classifiers,filename, columns=False):
    arr_scores_params=np.empty(shape=(len(classifiers),len(keys)), dtype=object)
    for key in range(len(keys)):
        for classifier in range(len(classifiers)):
            if columns:
                X_train, X_test, y_train, y_test, dict_labels=col_load(keys[key])
            else:
                X_train, X_test, y_train, y_test, dict_labels=load(keys[key], samples)
            ser_scores_params, df_cm, report, clf=classify(classifiers[classifier], X_train, y_train, X_test, y_test, dict_labels, keys[key])
            arr_scores_params[classifier, key]=ser_scores_params
    make_file(arr_scores_params, keys, classifiers, filename)

def uniprotlld_test(samples, classifiers,columns=False):
    arr_scores_params=np.empty(shape=(len(classifiers)), dtype=object)
    X_train_lld, y_train_lld, X_test_uni, y_test_uni, dict_labels=relabel(uniprot, lld, samples,columns=columns)
    for classifier in range(len(classifiers)):
        ser_scores_params, df_cm, report, clf=classify(classifiers[classifier], X_train_lld, y_train_lld, X_test_uni, y_test_uni, dict_labels, 'train lld, test uniprot')
        arr_scores_params[classifier]=ser_scores_params
    make_file(arr_scores_params, [0], classifiers, 'measurement_uniprot_lld.xlsx')
    return arr_scores_params

def bdp_test(samples, keys,classifiers,columns=False):
    arr_scores_params=np.empty(shape=(len(classifiers),len(keys)), dtype=object)
    for key in range(len(keys)):
        for classifier in range(len(classifiers)):
            gc.collect()
            X_train, y_train, X_test, y_test, dict_labels=smash_together(keys, keys[key], samples, columns=columns)
            ser_scores_params, df_cm, report, clf=classify(classifiers[classifier], X_train, y_train, X_test, y_test, dict_labels, 'train {}, test {}'.format(keys, keys[key]))
            arr_scores_params[classifier, key]=ser_scores_params
    make_file(arr_scores_params, keys, classifiers, 'measurement_bdp_test.xlsx')

def relabel(key, key_b, samples, columns=False):
    if columns:   
        _, X_test, _, y_test, dict_labels=col_load(uniprot)
        X_train_b, _, y_train_b, _, dict_labels_b=col_load(lld)
    else:
        _, X_test, _, y_test, dict_labels=load(uniprot, samples)
        X_train_b, _, y_train_b, _, dict_labels_b=load(lld, samples)
    
    ar_props = np.array(list(dict_labels.items()))[:,1]
    ar_props_b = np.array(list(dict_labels_b.items()))[:,1]

    #define labels 
    dict_prop_uniq = dict(enumerate(ar_props_b))
    dict_prop_uniq_rev = {value:key for key, value in dict_prop_uniq.items()}
    
    #finding all uniprot dt properties that are present in lld
    intersect=list(np.intersect1d(ar_props, ar_props_b))
    intersect_idx=[dict_prop_uniq_rev[prop] for prop in list(intersect)]
    dict_intersect = dict(zip(intersect_idx, intersect))

    #relabeling of uniprot labels if in intersection
    y_test=[dict_labels[prop] for prop in list(y_test)]
    y_test_is=[dict_prop_uniq_rev[prop] for prop in y_test if prop in list(dict_intersect.values())]
    y_test_idx=[prop for prop in range(len(y_test)) if y_test[prop] in list(dict_intersect.values())]
    
    #exclude dt properties which are not in the intersection
    X_test_is=X_test[y_test_idx,:]
    
    return  X_train_b, y_train_b, X_test_is, y_test_is, dict_prop_uniq

def smash_together(train_key_list, test_key, samples, columns=False):
    if columns:
        _, X_test, _, y_test, dict_labels_test=col_load(test_key)
    else:
        _, X_test, _, y_test, dict_labels_test=load(test_key, samples)
    arr_train=np.empty(shape=(len(train_key_list), 3), dtype=object)
    for key in range(len(train_key_list)):
        if columns:
            X_train, _, y_train, _, dict_labels=col_load(train_key_list[key])
        else:
            X_train, _, y_train, _, dict_labels=load(train_key_list[key], samples)
        arr_train[key, 0]=X_train
        arr_train[key, 1]=y_train.astype(int)
        arr_train[key, 2]=np.array(list(dict_labels.values()))
    
    #finding all unique dt_properties
    ar_labels_uniq=np.unique(np.hstack(tuple((arr_train[i, 2] for i in range(arr_train.shape[0])))))
    dict_labels_uniq = dict(enumerate(ar_labels_uniq.flatten()))
    dict_labels_uniq_rev = {value:key for key, value in dict_labels_uniq.items()}

    #relabeling of test labels
    y_test=[dict_labels_test[prop] for prop in list(y_test)]
    y_test=np.array([dict_labels_uniq_rev[prop] for prop in y_test])
    
    #relabel train labels 
    for key in range(len(train_key_list)):
         arr_train[key, 1]=[arr_train[key, 2][prop] for prop in list(arr_train[key, 1])]
         arr_train[key, 1]=[dict_labels_uniq_rev[prop] for prop in arr_train[key, 1]]
         
    #vstack and shuffle trainingsdata
    X_train=vstack(tuple((arr_train[i, 0] for i in range(arr_train.shape[0]))))
    y_train=np.hstack(tuple((arr_train[i, 1] for i in range(arr_train.shape[0]))))
    index = np.arange(np.shape(X_train)[0])
    np.random.seed(42)
    np.random.shuffle(index)
    X_train=X_train[index,:]
    y_train=y_train[index]
    return X_train, y_train, X_test, y_test, dict_labels_uniq, 

def alpha_num_separation(key, samples, alpha=True, columns=False):
    if columns:
        X_train, X_test, y_train, y_test, dict_labels=col_load(key)
    else:   
        X_train, X_test, y_train, y_test, dict_labels=load(key, samples)
    df=pd.read_excel(r'data/{}_dt_properties.xlsx'.format(key),engine='openpyxl',parse_dates=True)
    _, dt=get_datatype_stats(df, nl_tresh=0.5)

    rev_dict_labels={v: k for k, v in dict_labels.items()}
    dt_code=dt.where(dt=='code').dropna()
    dt_nl=dt.where(dt=='language').dropna()
    dt_alpha=pd.concat([dt_code,dt_nl])
    dt_num=dt.drop(labels=list(dt_alpha.keys()))
    
    y_train_alpha=[dict_labels[label] for label in y_train]
    train_idx_alpha = [idx for idx, element in enumerate(y_train_alpha) if element in dt_alpha]
    y_train_alpha=pd.Series(y_train_alpha)[train_idx_alpha]
    y_train_alpha=[rev_dict_labels[label] for label in y_train_alpha]
    X_train_alpha=X_train[train_idx_alpha,:]
    
    y_test_alpha=[dict_labels[label] for label in y_test]
    test_idx_alpha = [idx for idx, element in enumerate(y_test_alpha) if element in dt_alpha]
    y_test_alpha=pd.Series(y_test_alpha)[test_idx_alpha]
    y_test_alpha=[rev_dict_labels[label] for label in y_test_alpha]
    X_test_alpha=X_test[test_idx_alpha,:]
    
    y_train_num=[dict_labels[label] for label in y_train]
    train_idx_num = [idx for idx, element in enumerate(y_train_num) if element in dt_num]
    y_train_num=pd.Series(y_train_num)[train_idx_num]
    y_train_num=[rev_dict_labels[label] for label in y_train_num]
    X_train_num=X_train[train_idx_num,:]
    
    y_test_num=[dict_labels[label] for label in y_test]
    test_idx_num = [idx for idx, element in enumerate(y_test_num) if element in dt_num]
    y_test_num=pd.Series(y_test_num)[test_idx_num]
    y_test_num=[rev_dict_labels[label] for label in y_test_num]
    X_test_num=X_test[test_idx_num,:]
    
    if alpha:
        return X_train_alpha, y_train_alpha, X_test_alpha, y_test_alpha, dict_labels
    else:
        return X_train_num, y_train_num, X_test_num, y_test_num, dict_labels

def alpha_num_test(samples, keys,classifiers,columns=False, alpha=True, filename='cell_alpha_measurement.xlsx'):
    arr_scores_params=np.empty(shape=(len(classifiers),len(keys)), dtype=object)
    for key in range(len(keys)):
        for classifier in range(len(classifiers)):
            X_train, y_train, X_test, y_test, dict_labels=alpha_num_separation(keys[key], samples, alpha=alpha, columns=columns)
            ser_scores_params, df_cm, report, clf=classify(classifiers[classifier], X_train, y_train, X_test, y_test, dict_labels, keys[key])
            arr_scores_params[classifier, key]=ser_scores_params
    make_file(arr_scores_params, keys, classifiers, filename)
    

def mrr_score(y_test, y_prob):
    y_prob=np.argsort(-y_prob, axis=1)
    return np.mean([1/(np.where(y_prob[i,:]==y_test[i])[0][0]+1) for i in range(y_prob.shape[0])])   

if __name__=='__main__':
    uniprot='uniprot'
    lld='lld'
    key='uniprot'
    keys=['uniprot','lld', 'old','nbdc', 'dbpedia', 'wikidata', 'sherlock', 'city']
    keys=['uniprot','lld','old','nbdc','dbpedia','wikidata']
    train_keys=['uniprot','lld','old','nbdc']
    samples=100 #1000
    classifier='RF' #RF
    classifiers=['RF','LinSVM','KNN', 'GNB', 'LR'] #GBoost, #SVM scales bad, LinSVM does not have predict_proba
    columns=False

    #sample classification for CeLa
    '''
    columns=False
    X_train, X_test, y_train, y_test, dict_labels=load(key, samples)
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key)
    '''
    
    #sample classification for CoLa
    
    columns=True
    X_train, X_test, y_train, y_test, dict_labels=col_load(key) 
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key)
    
    
    #sample classification for lld uniprot overlap
    ''''
    X_train_lld, y_train_lld, X_test_uni, y_test_uni, dict_labels=relabel(uniprot, lld, samples, columns=columns)
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train_lld, y_train_lld, X_test_uni, y_test_uni, dict_labels, 'train lld, test uniprot')
    '''
    
    #sample classification for training on all biomedical databases
    '''
    X_train, y_train, X_test, y_test, dict_labels=smash_together(train_keys, key, samples, columns=columns)
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, 'mix')
    '''
    
    #run all classification measurements
    '''
    keys=['uniprot','lld', 'old','nbdc', 'dbpedia', 'wikidata', 'sherlock', 'city']
    main(samples, keys,classifiers,columns=columns)
    uniprotlld_test(samples, classifiers,columns=columns)
    bdp_test(samples, train_keys,classifiers,columns=columns)
    '''
    
    #alpha num tests
    '''
    keys=['uniprot','lld','old','nbdc','dbpedia','wikidata']
    #alpha_num_test(samples, keys,classifiers,columns=False, alpha=True, filename='cell_alpha_measurement.xlsx')
    #alpha_num_test(samples, keys,classifiers,columns=False, alpha=False, filename='cell_num_measurement.xlsx')
    alpha_num_test(samples, keys,classifiers,columns=True, alpha=True, filename='col_alpha_measurement.xlsx')
    alpha_num_test(samples, keys,classifiers,columns=True, alpha=False, filename='col_num_measurement.xlsx')
    '''