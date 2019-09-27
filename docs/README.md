

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
     
#### Attribution & Licensing

This derivative work employs data from the Seshat Databank (seshatdatabank.info) under Creative Commons Attribution Non-Commercial (CC By-NC SA) licensing.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Whitehouse, H., P. François, P.E. Savage, […] P. Turchin. “Complex societies precede moralizing gods throughout world history.” Nature. http://doi.org/10.1038/s41586-019-1043-4.

Turchin, P., T. E. Currie, H. Whitehouse, P. François, K. Feeney,  […] C. Spencer. 2017. “Quantitative historical analysis uncovers a single dimension of complexity that structures global variation in human social organization.” PNAS. http://www.pnas.org/content/early/2017/12/20/1708800115.full.

Turchin, P. 2018. “Fitting Dynamic Regression Models to Seshat Data.” Cliodynamics 9(1): 25-58. https://doi.org/10.21237/C7clio9137696.

Turchin, P., R. Brennan, T. E. Currie, K. Feeney, P. François, […] H. Whitehouse. 2015. “Seshat: The Global History Databank.” Cliodynamics 6(1): 77–107. https://doi.org/10.21237/C7clio6127917.

The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official positions, either expressed or implied, of the Seshat Databank, its collaborative scholarly community, or the Evolution Institute.
