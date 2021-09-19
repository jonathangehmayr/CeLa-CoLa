import numpy as np
import pandas  as pd
import string
import time
import os
import nltk
import random
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.stats import entropy, mode, kurtosis, skew
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MaxAbsScaler
from helpers import unicodeToAscii, representsDate, representsInt, representsFloat
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from helpers import drop_nan_inf_none, check_float_min_max, shape_for_scale, shape_for_training, printables, dict_char_classes, global_stats
from sherlock.features.preprocessing import convert_string_lists_to_lists #needs to be installed from https://github.com/mitmedialab/sherlock-project


def extract_global_stats(col):
    ser_global_stats=pd.Series(dtype=float)
    
    #percentage count only has None values
    ser_global_stats['n_none']=np.sum(col.isna())
    ser_global_stats['frac_none']=np.sum(col.isna())/len(col)
    col=drop_nan_inf_none(col)

    #entropy feature
    value,counts = np.unique(col.to_numpy(), return_counts=True)
    ser_global_stats['entropy']=entropy(counts)
    
    #number unique values
    ser_global_stats['frac_unique']=col.nunique()/len(col)
    
    #fraction of cells with numerical, alphabetic and symbolic chars
    ser_global_stats['frac_num']=np.sum(col.str.contains('[0-9]', regex=True))/len(col)
    ser_global_stats['frac_alpha']=np.sum(col.str.contains('[a-z]|[A-Z]', regex=True))/len(col)
    ser_global_stats['frac_sym']=np.sum(col.str.contains('[{}]'.format(dict_char_classes['special']), regex=True))/len(col)
    
    #mean and std of numerical alphabetic and symbolic chars in column
    ser_global_stats['mean_num_count']=np.mean(col.str.count('[0-9]'))
    ser_global_stats['std_num_count']=np.std(col.str.count('[0-9]'))
    ser_global_stats['mean_alpha_count']=np.mean(col.str.count('[a-z]|[A-Z]'))
    ser_global_stats['std_alpha_count']=np.std(col.str.count('[a-z]|[A-Z]'))
    ser_global_stats['mean_sym_count']=np.mean(col.str.count('[{}]'.format(dict_char_classes['special'])))
    ser_global_stats['std_sym_count']=np.std(col.str.count('[{}]'.format(dict_char_classes['special'])))
        
    #cell length stats
    n_chars_per_cell=[len(cell) for cell in col.tolist()]
    ser_global_stats['sum_chars']=np.sum(n_chars_per_cell)
    ser_global_stats['min_chars']=np.min(n_chars_per_cell)
    ser_global_stats['max_chars']=np.max(n_chars_per_cell)
    ser_global_stats['median_chars']=np.median(n_chars_per_cell)
    ser_global_stats['mode_chars']=mode(n_chars_per_cell).mode[0]
    ser_global_stats['kurt_chars']=kurtosis(n_chars_per_cell)
    ser_global_stats['skew_chars']=skew(n_chars_per_cell)
    ser_global_stats['any_chars']=np.any(n_chars_per_cell)
    ser_global_stats['all_chars']=np.all(n_chars_per_cell)

    #additional:
    ser_global_stats['dates']=np.sum([int(representsDate(cell)) for cell in col.tolist()])
    numbs=[float(cell) for cell in col.tolist() if representsInt(cell) | representsFloat(cell)]
    numbs=list(drop_nan_inf_none(pd.Series(numbs,dtype=float)))
    if len(numbs)>0: 
        ser_global_stats['min_num']=np.min(numbs) #24
        ser_global_stats['max_num']=np.max(numbs)
        ser_global_stats['median_num']=np.median(numbs)
        ser_global_stats['mean_num']=np.mean(numbs)
        ser_global_stats['mode_num']=mode(numbs).mode[0]
        ser_global_stats['std_num']=np.std(numbs)
        ser_global_stats['kurt_num']=kurtosis(numbs)
        ser_global_stats['skew_num']=skew(numbs)
    else:
        ser_global_stats['min_num']=-1
        ser_global_stats['max_num']=-1
        ser_global_stats['median_num']=-1
        ser_global_stats['mean_num']=-1
        ser_global_stats['mode_num']=-1
        ser_global_stats['std_num']=-1
        ser_global_stats['kurt_num']=-1
        ser_global_stats['skew_num']=-1
    return ser_global_stats

def extract_char_dist(col):
    col=drop_nan_inf_none(col)
    arr_char_dist=np.zeros(shape=(len(col), len(printables)))
    for row in range(len(col)):
        for char in col[row]:
            arr_char_dist[row,printables.index(char)]+=1
    ser_char_dist=pd.Series(np.mean(arr_char_dist, axis=0), dtype=float)
    return ser_char_dist

def extract_word_shape(col):
    col=drop_nan_inf_none(col)
    arr_word_shape=np.zeros(shape=(len(col), len(printables)))
    for row in range(len(col)):
        char_counter=0
        for char in col[row]:
            if char in dict_char_classes['lower']:
                arr_word_shape[row, char_counter]=1
            if char in dict_char_classes['upper']:
                arr_word_shape[row, char_counter]=2
            if char in dict_char_classes['number']:
                arr_word_shape[row, char_counter]=3
            if char in dict_char_classes['special']:
                arr_word_shape[row, char_counter]=4
            char_counter+=1
            if char_counter==n_chars-1:
                break
    ser_word_shape=pd.Series(mode(arr_word_shape, axis=0).mode[0], dtype=float)
    return ser_word_shape

def extract_word_embedding(col, gen_glove, dict_glove):
    col=drop_nan_inf_none(col)
    col_vecs=np.zeros(shape=(len(col), len(list(dict_glove.values())[0])), dtype=float)
    list_word_vecs=[]
    for row in range(len(col)):
        words=nltk.word_tokenize(col[row])
        words=[word.lower() for word in words]
        print(words)
        word_vecs=np.array([dict_glove[word] for word in words if word in gen_glove], dtype=float)
        print(word_vecs)
        list_word_vecs.append(word_vecs)
        if word_vecs.size>0:
            col_vecs[row]=np.mean(word_vecs, axis=0)
        ser_word_embedding=pd.Series(np.mean(col_vecs, axis=0), dtype=float)
    return ser_word_embedding    
    
def get_glove_dict():
    dict_glove = {}
    with open('data/glove/glove.6B.50d.txt','r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], float)
            dict_glove[word] = vector
    return dict_glove

def rearrange_df(df):
    dict_labels = dict(enumerate(column.strip() for column in df.columns))
    dict_rev_labels = {value:key for key, value in dict_labels.items()}
    row_step=20
    df_=pd.DataFrame()
    for column in df.columns: #delete index
        for row in range(0,len(df),row_step):
            col=df[column][row:row+row_step].reset_index(drop=True)
            col=pd.Series([unicodeToAscii(str(cell)) for cell in col.tolist()],dtype=str)
            df_.insert(len(df_.columns),dict_rev_labels[column], col, allow_duplicates=True)
    ser_labels=pd.Series(dict_labels)
    return df_, ser_labels
    
def make_tdocs(df, lists=False):
    df.columns=list(range(0,len(df.columns)))
    for i in df: #change to df[df.columns[0:3]]
        if lists:
            col=df[i][0]
        else:
            col=df[i]
        words=[token for j in range(len(col)) for token in nltk.word_tokenize(col[j])]
        tdoc=TaggedDocument(words, [i])
        yield tdoc

def train_doc2vec_model(df_train, key, n_chars, lists=False):
    train_corpus=list(make_tdocs(df_train, lists=lists))
    model = Doc2Vec(dm=0, workers=4, vector_size=n_chars, epochs=40, min_count=2, seed=42)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model_name='data/doc2vec/{}_doc2vec_model'.format(key)
    model.save(model_name)

def extract_paragraph_vec(col, key):
    col=drop_nan_inf_none(col)
    model_name='data/doc2vec/{}_doc2vec_model'.format(key)
    model = Doc2Vec.load(model_name)
    words=[token for i in range(len(col)) for token in nltk.word_tokenize(col[i])]
    ser_paragraph_vec = pd.Series(model.infer_vector(words), dtype=float)
    return ser_paragraph_vec

def extract_histogram(col):
    col=drop_nan_inf_none(col)
    hist=col.value_counts(ascending=True)#ascending=True
    ser_hist=pd.Series(resample(hist, 20),dtype=float)
    return ser_hist

def feature_extraction(df_,key, n_chars, n_features, lists=False):
    dict_glove=get_glove_dict()
    gen_glove=(key for key in dict_glove.keys())
    y_=pd.Series(list(df_.columns))
    df_.columns=list(range(0,len(df_.columns)))
    col_counter=1
    df_feat=pd.DataFrame(columns=['arrays', 'labels'])
    for col in df_: 
        if lists:
            column=pd.Series(df_[col][0])
        else:
            column=df_[col]
        if len(drop_nan_inf_none(column))>0:
            ser_global_stats=extract_global_stats(column)
            ser_char_dist_stats=extract_char_dist(column)
            ser_word_shape_stats=extract_word_shape(column)
            ser_glove_embedding=extract_word_embedding(column, gen_glove, dict_glove)
            ser_paragraph_vec=extract_paragraph_vec(column, key)
            ser_hist=extract_histogram(df_[col])

            arr_feat=np.zeros(shape=(n_features,n_chars))
            arr_feat[0,0:len(ser_global_stats)]=ser_global_stats
            arr_feat[1,0:len(ser_char_dist_stats)]=ser_char_dist_stats
            arr_feat[2,0:len(ser_word_shape_stats)]=ser_word_shape_stats
            arr_feat[3,0:len(ser_glove_embedding)]=ser_glove_embedding
            arr_feat[4,0:len(ser_paragraph_vec)]=ser_paragraph_vec
            arr_feat[5,0:len(ser_hist)]=ser_hist
            
            
            label=y_[col]
            ser_feat=pd.Series([arr_feat, label], index=['arrays', 'labels'])
            df_feat=df_feat.append(ser_feat, ignore_index=True)
        else:
            continue
        print('extraction: {} / {}'.format(col_counter, len(df_.columns)))
        col_counter+=1
    return df_feat

def preprocess(df_train, df_test):
    y_train = df_train[df_train.columns[1]].to_numpy().astype(int)
    X_train = df_train[df_train.columns[0]].to_numpy()
    y_test = df_test[df_test.columns[1]].to_numpy().astype(int)
    X_test = df_test[df_test.columns[0]].to_numpy()
    
    #checking if the float64 value constraints still apply
    X_train=check_float_min_max(X_train)
    X_test=check_float_min_max(X_test)
    
    #seperate the first row of feature vectors as it needs to be scaled
    #differently than the other rows
    X_train_single=np.zeros(shape=(X_train.shape[0],X_train[0].shape[1]))
    for i in range(len(X_train)):
        X_train_single[i,:]=X_train[i][0,:]
    
    X_train_other=np.zeros(shape=(X_train.shape[0]), dtype=object)
    for i in range(len(X_train)):
        X_train_other[i]=X_train[i][1:n_features,:]
        
    X_test_single=np.zeros(shape=(X_test.shape[0],X_test[0].shape[1]))
    for i in range(len(X_test)):
        X_test_single[i,:]=X_test[i][0,:]
    
    X_test_other=np.zeros(shape=(X_test.shape[0]), dtype=object)
    for i in range(len(X_test)):
        X_test_other[i]=X_test[i][1:n_features,:]
        
    #bring the feature rows in another shape for scaling
    X_train_shape=(len(X_train), X_train[0].shape[1])
    X_test_shape=(len(X_test), X_test[0].shape[1])
    X_train_other=shape_for_scale(X_train_other, X_train_shape, n_features-1)
    X_test_other=shape_for_scale(X_test_other, X_test_shape, n_features-1)
    
    #scale single feature and row features seperately
    Scaler = MaxAbsScaler()
    X_train_other= Scaler.fit_transform(X_train_other)
    X_test_other = Scaler.transform(X_test_other)

    Scaler = MaxAbsScaler()
    X_train_single = Scaler.fit_transform(X_train_single)
    X_test_single = Scaler.transform(X_test_single)
    
    #transform row feature back to original shape and add single features
    X_train_other=shape_for_training(X_train_other, X_train_shape, n_features-1)
    X_test_other=shape_for_training(X_test_other, X_test_shape, n_features-1)

    X_train=np.zeros(shape=(X_train_shape[0], n_features*X_train_shape[1]))
    for i in range(X_train_shape[0]):
        X_train[i, 0:X_train_single.shape[1]]=X_train_single[i,:]
        X_train[i, X_train_single.shape[1]:]=X_train_other[i,:]

    X_test=np.zeros(shape=(X_test_shape[0], n_features*X_test_shape[1]))
    for i in range(X_test_shape[0]):
        X_test[i, 0:X_test_single.shape[1]]=X_test_single[i,:]
        X_test[i, X_test_single.shape[1]:]=X_test_other[i,:]
    
    #deleting additional paddings
    idx1=list(range(31,95))
    idx2=list(range(335,380))
    idx3=list(range(495,570))
    idx=idx1+idx2+idx3
    
    X_train=np.delete(X_train, idx, 1)
    X_test=np.delete(X_test, idx, 1)

    #X and y are put together in one matrix, shuffled and made sparse
    X_train=np.append(X_train, y_train.reshape(-1, 1), axis=1)
    X_test=np.append(X_test, y_test.reshape(-1, 1), axis=1)

    X_train=shuffle(X_train, random_state=42)
    X_test=shuffle(X_test, random_state=42)

    X_train = csr_matrix(X_train)
    X_test = csr_matrix(X_test)
    return X_train, X_test

def save(X_train, X_test, ser_labels, key):
    print('saving...')
    save_npz(r'data/extracted_features/col_{}/{}_train_data.npz'.format(key, key), X_train)
    save_npz(r'data/extracted_features/col_{}/{}_test_data.npz'.format(key, key), X_test)
    
    file_name=r'data/extracted_features/col_{}/{}_properties.xlsx'.format(key, key)
    with pd.ExcelWriter(file_name) as writer:
        ser_labels.to_excel(writer, index=False)
    print('saved!')     

def load(key):
    print('loading data...')
    X_train = load_npz(r'data/extracted_features/col_{}/{}_train_data.npz'.format(key, key))
    X_test = load_npz(r'data/extracted_features/col_{}/{}_test_data.npz'.format(key, key))
    dict_lables=pd.read_excel(r'data/extracted_features/col_{}/{}_properties.xlsx'.format(key, key), engine='openpyxl').to_dict()[0]
    y_train=np.ravel(X_train[:,-1].todense())
    X_train=X_train[:,:-1]
    y_test=np.ravel(X_test[:,-1].todense())
    X_test=X_test[:,:-1]
    print('loaded data!')
    return X_train, X_test, y_train, y_test, dict_lables

def load_sherlock_data():
    data = pd.read_parquet("data/sherlock/test_values.parquet").reset_index(drop=True)
    labels = pd.read_parquet("data/sherlock/test_labels.parquet").reset_index(drop=True)  
    data=data.T.squeeze()
    labels=labels.T.squeeze()
    labels_unique=labels.unique()
    ser_X=pd.Series(dtype=object)
    ser_y=pd.Series(dtype=object)
    for label in labels_unique: #change index
        idx_labels=list(labels[labels==label][:50].index)
        list_labels=list(labels[labels==label][:50].values)
        data_cols=data[idx_labels]
        ser_X=pd.concat([ser_X, data_cols]).reset_index(drop=True)
        ser_y=pd.concat([ser_y, pd.Series(list_labels)]).reset_index(drop=True)
    
    ser_X, ser_y=convert_string_lists_to_lists(ser_X, ser_y)
    ser_X=pd.Series([[unicodeToAscii(str(item)) for item in random.sample(ser_X[i], min(len(ser_X[i]),1000))] for i in range(len(ser_X))]) 
    dict_labels = dict(zip(set(ser_y), range(0,len(set(ser_y)))))
    ser_y=pd.Series([dict_labels[label] for label in ser_y]) 
    ser_X_train, ser_X_test, ser_y_train, ser_y_test=train_test_split(ser_X, ser_y, test_size=0.2, random_state=42)
    df_train = pd.DataFrame(ser_X_train).transpose()
    df_train.columns=ser_y_train
    df_test = pd.DataFrame(ser_X_test).transpose()
    df_test.columns=ser_y_test
    ser_labels=pd.Series(dict([(value, key) for key, value in dict_labels.items()]))
    return df_train, df_test, ser_labels

def load_city_data(range_start, range_end):
    data=[]
    labels=[]
    for i in range(range_start,range_end):
        file_path='data/city/s{}.txt'.format(i)
        f=open(file_path)
        source=f.readlines()
        y=[item for item in source if item[0] in string.ascii_lowercase and ':' in item]
        labels=labels+y
        indices=[i for i in range(len(source)) if source[i][0] in string.ascii_lowercase and ':' in source[i]]
        for idx in indices:
            prop=[unicodeToAscii(item.split(' ', 1)[1]) for item in source[idx+2 : idx+2 + int(source[idx+1])]]
            data.append(prop)

    ser_X=pd.Series(data)
    ser_y=pd.Series(labels)
    dict_labels = dict(zip(set(ser_y), range(0,len(set(ser_y)))))
    ser_y=pd.Series([dict_labels[label] for label in ser_y])
    
    df_ = pd.DataFrame(ser_X).transpose()
    df_.columns=ser_y
    ser_labels=pd.Series(dict([(value, key) for key, value in dict_labels.items()]))
    return df_, ser_labels
    

if __name__=='__main__':
    keys=['uniprot', 'lld', 'old', 'nbdc', 'dbpedia', 'wikidata', 'city', 'sherlock']
    key='uniprot'
    n_chars=95 #includes 99% of whole cells for uniprot
    n_features=6
    tic = time.perf_counter() 
    
    #feature extraction for uniprot, lld, old, nbdc, dbpedia and wikidata
    '''
    for key in keys:
        df=pd.read_excel(r'data/{}_dt_properties.xlsx'.format(key),engine='openpyxl',parse_dates=True)
        df_train, df_test=train_test_split(df, test_size=0.2, random_state=42)
        df_train, ser_labels=rearrange_df(df_train)
        df_test, ser_labels=rearrange_df(df_test)
        train_doc2vec_model(df_train, key, n_chars)
        df_train=feature_extraction(df_train, key, n_chars, n_features)
        df_test=feature_extraction(df_test, key, n_chars, n_features)
        X_train, X_test=preprocess(df_train, df_test)
        save(X_train, X_test, ser_labels, key)
    ''' 
    
    #sherlock feature extraction
    '''
    key='sherlock'
    df_train, df_test, ser_labels=load_sherlock_data()
    train_doc2vec_model(df_train, key, n_chars, lists=True)
    df_train=feature_extraction(df_train, key, n_chars, n_features, lists=True)
    df_test=feature_extraction(df_test, key, n_chars, n_features, lists=True)
    X_train, X_test=preprocess(df_train, df_test)
    save(X_train, X_test, ser_labels, key)
    X_train, X_test, y_train, y_test, dict_lables=load(key)
    '''

    #city feature extraction
    '''
    key='city'
    df_train, ser_labels=load_city_data(1, 6)
    df_test, ser_labels=load_city_data(6, 11)
    train_doc2vec_model(df_train, key, n_chars, lists=True)
    df_train=feature_extraction(df_train, key, n_chars, n_features, lists=True)
    df_test=feature_extraction(df_test, key, n_chars, n_features, lists=True)
    X_train, X_test=preprocess(df_train, df_test)
    save(X_train, X_test, ser_labels, key)
    X_train, X_test, y_train, y_test, dict_lables=load(key)
    '''

    toc = time.perf_counter()
    duration=toc-tic
    print('duration: {}'.format(duration))
    

