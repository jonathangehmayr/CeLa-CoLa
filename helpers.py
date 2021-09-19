import pandas as pd
import numpy as np
import math
import unicodedata
from datetime import datetime

#len is 95 as DEL is excluded
printables='''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
dict_char_classes={
        'lower':'abcdefghijklmnopqrstuvwxyz',
        'upper':'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'number':'0123456789',
        'special':'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
        }

single_value_feats=['n_chars', 'n_numbers', 'n_letters', 'n_spaces',
                    'n_uppers', 'n_specials', 'is_date', 'perc_numbers',
                    'perc_letters', 'perc_specials', 'first_letter', 'digits_before_comma',
                    'digits_after_comma', 'negativ_number', 'first_number', 
                    'unit_place', 'tenth_place']

cell_feats=['char_dist', 'word_shape', 'dict_lookup']

global_stats=['n_none','frac_none', 'entropy','frac_unique','frac_num',
     'frac_alpha','frac_sym','mean_num_count', 'std_num_count',
     'mean_alpha_count', 'std_alpha_count', 'mean_sym_count', 
     'std_sym_count', 'sum_chars', 'min_chars','max_chars', 
     'median_chars', 'mode_chars','kurt_chars', 'skew_chars',
     'any_chars', 'all_chars','dates','min_num', 'max_num', 'median_num',
     'mean_num', 'mode_num', 'std_num', 'kurt_num', 'skew_num']

endpoint_dict =	{
  'uniprot':'http://sparql.uniprot.org/sparql', 
  'lld':'http://linkedlifedata.com/sparql', 
  'old':'http://sparql.openlifedata.org/', 
  'nbdc':'http://integbio.jp/rdf/sparql', 
  'dbpedia':'http://dbpedia.org/sparql',
  'wikidata': 'https://query.wikidata.org/sparql'
}


# from https://stackoverflow.com/a/518232/2809427
#normalizing the data to only contain ascii printables
def unicodeToAscii(s):
    chars=printables
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in chars)

#https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def representsInt(s):
    try: 
        int(s)
        return True
    except:
        return False
    
def representsFloat(s):
    try: 
        float(s)
        return True
    except:
        return False
    
 #from https://stackoverflow.com/questions/33204500/pandas-automatically-detect-date-columns-at-run-time
def representsDate(s):
    try:
        pd.to_datetime(s, utc=True)
        return True
    except:
        return False

def drop_nan_inf_none(col):
    col=pd.Series(list(filter(None, col)))
    for i in range(len(col)):
        try:
            cell=float(col[i])
            if np.isnan(cell):
                col=col.drop(i)
                continue
            if np.isinf(cell):
                col=col.drop(i)
                continue
            if cell>np.finfo(np.float32).max:
                col=col.drop(i)
                continue
        except:
            continue
    return col.reset_index(drop=True)

def check_float_min_max(X):
    for i in range(len(X)):
        mask=X[i]>np.finfo(np.float64).max
        X[i][mask]=np.finfo(np.float64).max
        mask=X[i]<np.finfo(np.float64).min
        X[i][mask]=np.finfo(np.float64).min
    return X

def shape_for_scale(X, X_shape, n_features):
    X_helper=np.zeros(shape=(X_shape[0]*X_shape[1], n_features))
    idx=0
    for i in range(X_shape[0]):
        for j in range(n_features):
            X_helper[idx:idx+X_shape[1],j]=X[i][j,:]
        idx=idx+X_shape[1]
    return X_helper

def shape_for_training(X, X_shape, n_features):
    shaped=np.zeros(shape=(X_shape[0], n_features*X_shape[1]))
    idxrow=0
    for i in range(X_shape[0]):
        idxcolumn=0
        for j in range(n_features):
            shaped[i,idxcolumn:idxcolumn+X_shape[1]]=X[idxrow:idxrow+X_shape[1],j]
            idxcolumn=idxcolumn+X_shape[1]
        idxrow=idxrow+X_shape[1]
    return shaped

def get_single_value_feat_rows(X, n_features):
    X_single_feat=np.zeros(shape=(X.shape[0], X[0].shape[1]))
    for i in range(len(X)):
        X_single_feat[i,:]=X[i][n_features-1,:]
    return X_single_feat

def add_scaled_single_value_feat_rows(X_single_feat_scaled, X_scaled, n_chars, n_features):
    idx_row=0
    for i in range(X_single_feat_scaled.shape[0]):
        X_scaled[idx_row:idx_row+n_chars, n_features-1]=X_single_feat_scaled[i]
        idx_row+=n_chars
    return X_scaled

def get_datatype_stats(df, nl_tresh=0.5):
    #detecting dates
    #from https://stackoverflow.com/questions/33204500/pandas-automatically-detect-date-columns-at-run-time
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
            except ValueError:
                pass
            except OverflowError:
                pass
    
    #detection of natural language
    dictionary = enchant.Dict("en_US")
    df_strings=df[df.columns[df.dtypes==object]]
    df_bool=pd.DataFrame(dtype=object)
    for i in range(len(df_strings.columns)):
        list_bool_row=[]
        for j in range(len(df_strings)):
            list_bool_cell=[]
            cell=df_strings[df_strings.columns[i]][j]
            if type(cell)==float:
                if math.isnan(cell):
                    cell='nan'
            try:
                words=nltk.word_tokenize(cell)
            except:
                words=[]
            words=[word for word in words if word.isalpha()]
            if words==[]:
                list_bool_cell.append(False)
            else:
                for k in range(len(words)):
                    is_word=dictionary.check(words[k])
                    list_bool_cell.append(is_word)        
            list_bool_row.append(list_bool_cell)            
        df_bool[df_strings.columns[i]]=list_bool_row
    
    series_perc_nl=pd.Series(dtype=float)
    
    for i in range(len(df_bool.columns)):
        l=0
        c=0
        for j in range(len(df_strings)):
            l+=len(df_bool[df_bool.columns[i]][j])
            c+=df_bool[df_bool.columns[i]][j].count(True)
        series_perc_nl[str(i)]=c/l
        
    dt=df.dtypes
    for column in df.columns:
        dt[column]=str(dt[column])
    
    df_stats=dt.value_counts(normalize=True)*len(df.columns)
    n_nl=sum(i >nl_tresh for i in series_perc_nl)
    
    for i in range(len(series_perc_nl)):
        if series_perc_nl[i]>=nl_tresh:
            dt[df_strings.columns[i]]='language'
        else:
            dt[df_strings.columns[i]]='code'
    
    idx='object'
    df_stats['code']=df_stats[idx]-n_nl
    df_stats['language']=n_nl
    df_stats=df_stats.drop(labels=[idx])
    df_stats=df_stats.sort_values(ascending=False)
    print('datatypes in table:')
    print(df_stats)
    print('different datatype properties: {}'.format(len(df.columns)))
    df_stats['different datatype properties']=len(df.columns)
    
    return df_stats, dt




