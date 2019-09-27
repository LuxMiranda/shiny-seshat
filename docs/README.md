

[![DOI](https://zenodo.org/badge/184840639.svg)](https://zenodo.org/badge/latestdoi/184840639)


#### [Click here for downloadable releases of Shiny Seshat](https://github.com/JackMiranda/shiny-seshat/releases)

## Improved and analysis-ready

Shiny Seshat is a scrubbed derivative of the [Seshat: Global History Databank](https://seshatdatabank.info) featuring:

*  Imputation of missing values via the DataWig deep neural network
*  Numeric, temporally-resolved data points 
*  Corrections of many human errors and typographical mistakes
*  And more!

This initial release of the dataset is intended as a Supplemental Material for the 2019 article _Social complexity clusters and recurrent social formations_ by Jack Miranda and Jacob Freeman. Please see the "Data and methods" section of the work for full information on the dataset, and cite appropriately if you use Shiny Seshat in your work!

#### Development setup

The program that scrubs the original Seshat datbank into Shiny Seshat is written in Python 3. If you wish to run it, first install these dependencies:

```
pip install numpy pandas datawig statsmodels hyperopt utm dill tqdm sklearn 
```

The process of fetching the original databank is automated. To download and begin scrubbing, simply run:

```
python scrub.py
```
     
#### License     
      
This work is licensed under Creative Commons Non-Commercial Share-Alike International 4.0

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
