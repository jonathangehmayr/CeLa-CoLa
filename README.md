# CeLa-CoLa
CeLa and CoLa are used for predicting labels for columns of semi structured data like tables. Cela and CoLa are two distinct feature engineering approaches, whereas the former extracts features for a single cell whereas the latter creates a feature vector for all values of a whole column.

The datasets used in this project were extracted using the dataset_creation.py file. Among the querried databases are:
-Wikidata
-DBpedia
-Uniprot
-LinkedLifeData
-OpenLifeData
-NBDC

The datasets are available in this repository in /data as .xlsx files.

Before using the code it is necessary to download the 50d GloVe dictionary and put it into the directory data/glove.

As the feature extraction process produces a lot of data it was not possible to upload the preextracted features with this non commercial account. The extracted features can be reporduced by using the files cola_feature_extraction.py and cela_feature_extraction.py. After extracting the features the classification.py is used to run multiple experiments. Among those Cola is trained on data from Pham (available here: https://github.com/minhptx/iswc-2016-semantic-labeling/tree/master/data/datasets/dbpedia/data) which is already included in this repo (data/city). To compare against the Sherlock model from Hulsebos go to the following link and copy test data into data/sherlock of this repo (https://github.com/mitmedialab/sherlock-project).
