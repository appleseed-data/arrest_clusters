# arrest_clusters analysis

This project processes arrest data from Chicago Police Department into an analysis-friendly format and runs several types of analysis. 


## Source

Based on data from the Chicago Data Portal. See disclaimer at bottom.

## Getting Started

1. Clone this repo

```terminal
git clone https://github.com/appleseed-data/arrest_clusters
```

2. Create a new conda environment from the environment.yml file. This is optional but highly recommended, but you can use your own env instead.

```terminal
conda env create -f environment.yml
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

### Data Prep

* TBP

  
### Data Classification of Arrest Charge Descriptions

* Direct Match on Text with Crosswalk
* Fuzzy Match on Text with Crosswalk
* NLP Match on Text with Crosswalk

## Analysis

* TBP

## Disclaimer

The data described herein and in associated stories are a derivative of the source from Chicago Data Portal. Per [the terms of use](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html) we provide the following dislcaimer:

*This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago.  The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site.  The data provided at this site is subject to change at any time.  It is understood that the data provided at this site is being used at one’s own risk.*

