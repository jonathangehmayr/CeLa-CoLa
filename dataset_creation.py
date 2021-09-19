from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import numpy as np
import time
from helpers import endpoint_dict

def get_dt_properties(key):
    dt_properties_query='''
    SELECT DISTINCT ?dp
    WHERE
    {
        ?dp a owl:DatatypeProperty.
    }
    '''
    sparql = SPARQLWrapper(endpoint_dict[key])
    sparql.setQuery(dt_properties_query)
    sparql.setReturnFormat(JSON)
    dt_properties_result = sparql.query().convert()

    dt_properties_list=[]
    
    for result in dt_properties_result['results']['bindings']:
        dt_properties_list.append(result['dp']['value'])
    return dt_properties_list
    
def query_for_dt_property_data(key, dt_properties_list, n):
    df_o= pd.DataFrame()
    df_s= pd.DataFrame()
    loops=len(dt_properties_list) #5 for testing
    for i in range(loops): 
        s_list=[] #is not used but maybe needed in future when subject is used for feature extraction too
        o_list=[]
        dt_property=dt_properties_list[i]    
        query='''
        SELECT ?s ?o
        WHERE
        {{
          ?s <{}> ?o
        }}
        ORDER BY RAND()
        LIMIT {}
        '''.format(dt_property,n)
        
        sparql = SPARQLWrapper(endpoint_dict[key])
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        error_counter=0
        error_lim=3
        while True:
            try:
                s_o_result = sparql.query().convert()
            except: #InternalServerError
                time.sleep(5)
                print('Error at {}'.format(i+1)) #+1 is just for printing the correct fraction
                error_counter+=1
                if error_counter==1:
                    break
                continue
            break
        
        if error_counter!=error_lim:
            for result in s_o_result["results"]["bindings"]:
                s_list.append(result["s"]["value"])
                o_list.append(result["o"]["value"])

            #saving None into o_list and s_list if no data or less then n data is available            
            olen=len(o_list)    
            if olen<n:
                if olen==0:
                    o_list=[None]*n
                    s_list=[None]*n
                    df_o[dt_property]=o_list
                    df_s[dt_property]=s_list
                else:
                    o_list_helper=o_list.copy()
                    o_list=[None]*n
                    s_list_helper=s_list.copy()
                    s_list=[None]*n
                    for j in range(len(o_list_helper)):
                       o_list[j]=o_list_helper[j]
                       s_list[j]=s_list_helper[j]
                    df_o[dt_property]=o_list
                    df_s[dt_property]=s_list
            else:
                df_o[dt_property]=o_list
                df_s[dt_property]=s_list
        else:
            o_list=[None]*n
            s_list=[None]*n
            df_o[dt_property]=o_list
            df_s[dt_property]=s_list
        
        print('Progress: {}/{}'.format(i+1,loops))
    return df_o, df_s

def get_subject_dt_properties(key, n, df_s):
    a = np.empty((n,len(df_s.columns)))
    a[:]=None
    df_s_dt_properties=pd.DataFrame(a,columns=df_s.columns,index=list(range(n)),dtype='object')
    for i in range(len(df_s.columns)):
        for j in range(n):
            dt_properties_list=[]
            subject_query='''
            SELECT DISTINCT ?dp 
            WHERE
            {{
              <{}> ?dp ?o.
              ?dp a owl:DatatypeProperty
            }}
            '''.format(df_s[df_s.columns[i]][j])
            
            sparql = SPARQLWrapper(endpoint_dict[key])
            sparql.setQuery(subject_query)
            sparql.setReturnFormat(JSON)
            error_lim=3
            error_counter=0
            while True:
                try:
                    subject_result = sparql.query().convert()
                except: #InternalServerError
                    time.sleep(5)
                    print('Error  at {}'.format((i*n+j)+1))
                    error_counter+=1
                    if error_counter==error_lim:
                        break
                    continue
                break
            
            for result in subject_result["results"]["bindings"]:
                dt_properties_list.append(result["dp"]["value"]) #change value to literal
                
            df_s_dt_properties[df_s.columns[i]][j]=dt_properties_list
            print('Progress: {}/{}'.format((i*n+j)+1, len(df_s.columns)*n))
    return df_s_dt_properties
    
def make_file(df,key):    
    file_name='{}_dt_properties.xlsx'.format(key)
    sheet_name='{}_datatype_properties'.format(key)
    with pd.ExcelWriter(file_name) as writer:  
        df.to_excel(writer, sheet_name=sheet_name,index=False)

def open_df(key):
    df=pd.read_excel(r'{}_dt_properties.xlsx'.format(key),engine='openpyxl')
    return df

def clean_data(df, tresh):
    #df=open_df(key)
    valid_data_list=df.count() #counts all none nan etc for each row
    empty_columns_list=valid_data_list[valid_data_list == 0].index.tolist()
    no_empty_columns_df=df.drop(empty_columns_list, axis=1)
    valid_data_list=no_empty_columns_df.count()
    few_data_columns_list=valid_data_list[valid_data_list < tresh].index.tolist()
    cleaned_df=no_empty_columns_df.drop(few_data_columns_list, axis=1)
    return cleaned_df    

def process_dt_properties(key,n, tresh, makeFile=False):
    dt_properties_list=get_dt_properties(key)
    df_o,df_s=query_for_dt_property_data(key, dt_properties_list, n)
    cleaned_df_o=clean_data(df_o, tresh)
    #cleaned_df_s=clean_data(df_s, tresh)
    #df_s_dt_properties=get_subject_dt_properties(key, n, cleaned_df_s)
    if makeFile:
        make_file(cleaned_df_o, key)
        #make_file(cleaned_df_s, key+'_subjects')
        #make_file(df_s_dt_properties, key+'_subjects_list')
    return cleaned_df_o, df_o, dt_properties_list#, cleaned_df_s, df_s_dt_properties
    
if __name__=='__main__':
    n=1000 #number of values for each datatype property
    tresh=0.8*n #treshhold for columns whose number of valid value is below n
    key='uniprot'
    keys=['uniprot','lld','old','nbdc','dbpedia','wikidata']
    cleaned_df_o, df_o, dt_properties_list=process_dt_properties(key,n, tresh, makeFile=True)