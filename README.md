# arrest_clusters analysis

This project processes arrest data from Chicago Police Department into an analysis-friendly format and runs several types of analysis. The current analytical focus is of a clustering nature; however, this project primarily manages a data cleansing and categorization process of Chicago arrest data. 


## Source

Based on data from the Chicago Data Portal. See disclaimer at bottom.

## Getting Started

1. Clone this repo

```terminal
git clone https://github.com/appleseed-data/arrest_clusters
```

2. Create a new conda environment from the environment.yml file. This is optional but highly recommended, but you can use your own env instead.

```terminal
conda env create --file environment.yml -n arrest_clusters
```

* Note, in some cases the conda env setup fails but does create an environment called arrest_clusters. If that's the case, activate the arrest_clusters environment and install dependencies with pip, see next. 

To switch conda envs:
```terminal
conda activate arrest_clusters
```

3. Install depedencies (recommend that you use the arrest_clusters env as noted above, but you can install this in any environment of your choosing.).

```terminal
pip install -r requirements.txt
```

4. From terminal, run the application

```terminal
python main.py
```

## Data

The primary output of this project is the classification of arrest records into a cleaned dataset that is ready for analysis. Currently the target output file is saved as a compressed Pandas Pickle file. The compression saves space and the pickle saves dtypes of each column such as categorical types and datetime types.

URL to Output file: 

How to Get and Read Compressed Pandas Pickle from a Git Repo
```python3
# get necessary dependencies
from io import BytesIO
import requests
import pandas as pd
import joblib

# link to the compressed pickle object in this repo
url = "https://github.com/appleseed-data/arrest_clusters/blob/main/data/arrests_redacted_classified.bz2?raw=true"

# function to get the pickled object
def get_git_pickle(data_path):
    """
    :params data_path: the text of a url string to the pickled object
    :return tgt_file: a pandas dataframe object
    """
    data_stream = BytesIO(requests.get(data_path).content)
    tgt_file = joblib.load(data_stream)
    return tgt_file

df = get_git_pickle(url)

print(df.head())
```


### Data Prep

A brief description of the data pipeline to process source arrest data. 

* TBP 

  
### Data Classification of Arrest Charge Descriptions

Overview. The CPD arrest data contains four columns of arrest charges, numbered from 1 to 4, i.e., Charge 1 Class, Charge 1 Description, etc. Not every arrest has charge data and not every arrest has more than one charge. Where an arrest record has charge information, a corresponding charge category description is created, i.e. Charge 1 Description Category Micro, Charge 1 Description Category Macro. The Micro category are classifications of a granular nature - about 40 categories of descriptions. The Macro category group the Micro categories into 6 high-level groups to aid in aggregate analysis.

* What: Convert raw text of arrest charge description to one of many discrete, semantic categories of arrest. 
* Why: There are thousands of charge descriptions that vary slightly but are fundamentally the same type of arrest charge, reducing the number of unique charges into discrete categories makes analysis simpler to perform. 
* How: Leverage a labeled dataset of charge descriptions to categories from this worksheet [CPD Crosswalk](https://github.com/appleseed-data/arrest_clusters/blob/main/data/CPD_crosswalk_final.xlsx) and apply classification methods.
* Other: The charges are also mapped to numeric representation of severity where a high number represents relative higher severity to other charges
#### Classification Methods to Map Charge Descriptions to Categories

Using a labeled dataset, map charge descriptions to various categories. Most records (approx 90%) are mapped through a direct or fuzzy match, the rest are mapped with an NLP classifier, n=523,184. 

* Direct Match on Text with Crosswalk: Where there is a direct match in the labeled dataset, map the associated charge categories to the charge description.
* Fuzzy Match on Text with Crosswalk: Where there is a slight (fuzzy) difference between the labeled dataset and the charge description (an edit distance of 4), map the charge description to a charge category.
* NLP Match on Text with Crosswalk: If any charges are not mapped after direct and fuzzy match (approx 10%), learn a model to peform multi-class text classification of charge descriptions to charge categories using Naive Bayes, then apply the model to claassify texts.

Confusion Matrix and Accuracy Information for NLP Classification of Charge Descriptions.
![confusion matrix 1](models/arrest_charge_descr_confusion_matrix.png)

Confusion Matrix and Accuracy Information for NLP Classification of Police-Related Flag where 0 is False, and 1 is True.
![confusion matrix 1](models/arrest_police_related_confusion_matrix.png)

## Analysis

* TBP

## Disclaimer

The data described herein and in associated stories are a derivative of the source from Chicago Data Portal. Per [the terms of use](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html) we provide the following dislcaimer:

*This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago.  The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site.  The data provided at this site is subject to change at any time.  It is understood that the data provided at this site is being used at one???s own risk.*

