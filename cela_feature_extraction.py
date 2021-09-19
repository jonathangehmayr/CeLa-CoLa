import numpy as np
import pandas  as pd
import math
import time
import sys
import nltk
from ast import literal_eval
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from helpers import unicodeToAscii, representsDate, representsFloat, representsInt, shape_for_scale, shape_for_training, get_single_value_feat_rows, add_scaled_single_value_feat_rows, printables, dict_char_classes, drop_nan_inf_none
from cola_feature_extraction import get_glove_dict

def feature_extraction(key, n_chars, samples, n_features):
    df=pd.read_excel(r'data/{}_dt_properties.xlsx'.format(key),engine='openpyxl',parse_dates=True)
    #df=df.iloc[:3,:5] #test
    dict_labels = dict(enumerate(column.strip() for column in df.columns))
    dict_rev_labels = {value:key for key, value in dict_labels.items()} 
    list_feat_names=['arrays', 'labels']
    df_feat=pd.DataFrame(columns=list_feat_names)
    cell_counter=1
    file_counter=0
    dict_glove=get_glove_dict()
    
    for column in df.columns:
        for row in range(len(df)):
            cells_len=len(df.columns)*samples
            cell=unicodeToAscii(str(df[column][row]))
            if cell is None:
                continue
            if not cell:
                continue
            try:
                cell=float(cell)
                if np.isnan(cell):
                    continue
                if np.isinf(cell):
                    continue
                cell=str(cell)
            except:
                pass
        
            #actual feature extraction
            arr_feat=np.zeros(shape=(n_features,n_chars))
            arr_idx=0  
            arr_feat,arr_idx=extract_char_dist_feat(cell, arr_feat, arr_idx)
            arr_feat, arr_idx=extract_word_shape_feat(cell, arr_feat, n_chars, arr_idx)
            arr_feat, arr_idx=extract_single_value_feat(cell, arr_feat, n_features, arr_idx)
            arr_feat, arr_idx=extract_word_embedding(cell, arr_feat, arr_idx, dict_glove)
            
            label=dict_rev_labels[column]
            ser_feat=pd.Series([arr_feat.tolist(), label], index=list_feat_names)
            df_feat=df_feat.append(ser_feat, ignore_index=True)
            print('extraction: {} / {}'.format(cell_counter, cells_len))
            
            file_name='{}_{}.xlsx'.format(key, file_counter)
            if cell_counter%100000==0:
                with pd.ExcelWriter(file_name) as writer:
                    df_feat.to_excel(writer, index=False)
                    df_feat=pd.DataFrame(columns=list_feat_names)
                    file_counter+=1
            cell_counter+=1
            
    with pd.ExcelWriter(file_name) as writer:
        df_feat.to_excel(writer, index=False)
        df_feat=pd.DataFrame(columns=list_feat_names)
        file_counter+=1
        
    list_df=[pd.read_excel(r'{}_{}.xlsx'.format(key, i),engine='openpyxl') for i in range(file_counter)]
    df_feat=pd.concat(list_df, ignore_index=True) 
    for i in range(df_feat.shape[0]):
        print(i)
        df_feat.iloc[i,0]=[np.array(literal_eval(df_feat.iloc[i,0]))]
    
    ser_labels=pd.Series(dict_labels) 
    return df_feat, ser_labels

def extract_single_value_feat(cell, arr_feat, n_features, arr_idx):
    list_feat=[]
    chars=len(cell)
    numbers = sum(char.isdigit() for char in cell)
    letters = sum(char.isalpha() for char in cell)
    spaces  = sum(char.isspace() for char in cell)
    uppers  = sum(char.isupper() for char in cell)
    others  = chars - numbers - letters - spaces
    date=int(representsDate(cell))
    
    #Chen
    perc_num=numbers/chars
    perc_let=letters/chars
    perc_sym=others/chars
    
    #Goel
    if not representsInt(cell) & representsFloat(cell):
        first_let=printables.index(cell[0])
    else:
        first_let=-1

    if representsInt(cell) | representsFloat(cell):
        first_num_idx=0
        no_num=True
        while(no_num):
            try:
                first_num=dict_char_classes['number'].index(cell[first_num_idx])
                no_num=False
            except ValueError:
                first_num_idx+=1
        if cell[0]=='-':
            negativ_num=1
        else:
            negativ_num=0
    else:
        first_num=-1 
        negativ_num=-1
        
    if representsFloat(cell):
        try:
            digits_before_comma=len(cell[0:cell.index('.')])
        except ValueError:
            digits_before_comma=-1
        try:
            digits_after_comma=len(cell[cell.index('.')+1:])
        except ValueError:
            digits_after_comma=-1
        try:
            unit_place=dict_char_classes['number'].index(cell[cell.index('.')+1])
        except ValueError:
            unit_place=-1
        try:
            tenth_place=dict_char_classes['number'].index(cell[cell.index('.')+2])
        except ValueError:
            tenth_place=-1
        except IndexError:
            tenth_place=-1
    else:
        digits_before_comma=-1
        digits_after_comma=-1
        unit_place=-1
        tenth_place=-1

    list_feat=[chars, numbers, letters, 
               spaces, uppers, others, 
               date, perc_num, perc_let, 
               perc_sym, first_let, digits_before_comma, 
               digits_after_comma, negativ_num, 
               first_num, unit_place, tenth_place]
    
    arr_feat[arr_idx, 0:len(list_feat)]=list_feat

    return arr_feat, arr_idx

def extract_char_dist_feat(cell, arr_feat, arr_idx):
    chars=printables
    for char in cell:
        arr_feat[arr_idx,chars.index(char)]+=1
    return arr_feat, arr_idx+1

def extract_word_shape_feat(cell, arr_feat, n_chars, arr_idx):
    dict_word_shape_classes=dict_char_classes
    char_counter=0
    for char in cell:
        if char in dict_word_shape_classes['lower']:
            arr_feat[arr_idx,char_counter]=1
        if char in dict_word_shape_classes['upper']:
            arr_feat[arr_idx,char_counter]=2
        if char in dict_word_shape_classes['number']:
            arr_feat[arr_idx,char_counter]=3
        if char in dict_word_shape_classes['special']:
            arr_feat[arr_idx,char_counter]=4
        char_counter+=1
        if char_counter==n_chars-1:
            break
    return arr_feat, arr_idx+1

def extract_word_embedding(cell, arr_feat, arr_idx, dict_glove):
    words=[word.lower() for word in nltk.word_tokenize(cell)]
    word_vec=np.array([dict_glove[word] for word in words if word in dict_glove], dtype=float)
    if word_vec.size>0:
        word_vec=np.mean(word_vec, axis=0)
    else:
        word_vec=np.zeros(shape=(50),dtype=float)
    arr_feat[arr_idx,:len(word_vec)]=word_vec
    return arr_feat, arr_idx+1

def preprocess_features(df_feat, n_features, n_chars):
    y = df_feat[df_feat.columns[1]].to_numpy().astype(int)
    X = df_feat[df_feat.columns[0]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_shape=(len(X_train), X_train[0].shape[1])
    X_test_shape=(len(X_test), X_test[0].shape[1])

    #scale single value feature rows seperately 
    X_train_single_feat=get_single_value_feat_rows(X_train, n_features)
    X_test_single_feat=get_single_value_feat_rows(X_test, n_features)
    Scaler = MaxAbsScaler()
    X_train_single_feat= Scaler.fit_transform(X_train_single_feat)
    X_test_single_feat= Scaler.transform(X_test_single_feat)
    
    #scale all features and add back seperately scaled single value features
    X_train=shape_for_scale(X_train, X_train_shape, n_features)
    X_test=shape_for_scale(X_test, X_test_shape, n_features)
    Scaler = MaxAbsScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)
    X_train=add_scaled_single_value_feat_rows(X_train_single_feat, X_train, n_chars, n_features)
    X_test=add_scaled_single_value_feat_rows(X_test_single_feat, X_test, n_chars, n_features)
    
    #produce sparse matrices for training
    X_train=shape_for_training(X_train, X_train_shape, n_features)
    X_test=shape_for_training(X_test, X_test_shape, n_features)
    
    #delete additional paddings
    idx1=list(range(207,285))
    idx2=list(range(335,380))
    idx=idx1+idx2
        
    X_train=np.delete(X_train, idx, 1)
    X_test=np.delete(X_test, idx, 1)
    
    X_train=np.append(X_train, y_train.reshape(-1, 1), axis=1)
    X_test=np.append(X_test, y_test.reshape(-1, 1), axis=1)
    
    X_train=csr_matrix(X_train)
    X_test=csr_matrix(X_test)
    
    print('preprocessed!')
    return X_train, X_test   

    
def save(X_train, X_test, ser_labels, key):
    print('saving...')
    save_npz(r'data/extracted_features/{}/{}_train_data.npz'.format(key, key), X_train)
    save_npz(r'data/extracted_features/{}/{}_test_data.npz'.format(key, key), X_test)
    file_name=r'data/extracted_features/{}/{}_properties.xlsx'.format(key, key)

    with pd.ExcelWriter(file_name) as writer:
        ser_labels.to_excel(writer, index=False)
    print('saved!')

def load(key, samples):
    print('loading data...')
    X_train = load_npz(r'data/extracted_features/{}/{}_train_data.npz'.format(key, key))
    X_test = load_npz(r'data/extracted_features/{}/{}_test_data.npz'.format(key, key))
    dict_labels=pd.read_excel(r'data/extracted_features/{}/{}_properties.xlsx'.format(key, key),engine='openpyxl').to_dict()[0]
    
    idx_train=int((X_train.shape[0]/800)* samples*0.8)
    idx_test=int((X_test.shape[0]/200)* samples*0.2)
    y_train=np.ravel(X_train[:idx_train,-1].todense())
    X_train=X_train[:idx_train,:-1]
    y_test=np.ravel(X_test[:idx_test,-1].todense())
    X_test=X_test[:idx_test,:-1]
    print('loaded data!')
    return X_train, X_test, y_train, y_test, dict_labels

if __name__=='__main__': 
    keys=['uniprot','lld','old','nbdc','dbpedia','wikidata']
    key='nbdc'
    samples=1000 #1000
    n_chars=95 #includes 98% of whole cells for uniprot
    n_features=4
    tic = time.perf_counter()
    
    #extract features for all databases
    '''
    for key in keys:
        df_feat, ser_labels=feature_extraction(key, n_chars, samples, n_features)    
        X_train, X_test,=preprocess_features(df_feat, n_features, n_chars)
        save(X_train, X_test, ser_labels, key)
        #X_train, X_test, y_train, y_test, dict_labels=load(key, samples)
    '''
    
    #extract features for one database
    '''
    df_feat, ser_labels=feature_extraction(key, n_chars, samples, n_features)    
    X_train, X_test,=preprocess_features(df_feat, n_features, n_chars)
    save(X_train, X_test, ser_labels, key)
    #X_train, X_test, y_train, y_test, dict_labels=load(key, samples)
    '''    

    toc = time.perf_counter()
    duration=toc-tic
    print('duration: {}'.format(duration))
