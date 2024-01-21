# Fog Climatology & Nowcasting

## Description
This project aims to improve the fog nowcasting capabilities at Dublin Airport by building classification models using machine learning and deep learning algorithms to predict fog at short lead times. The models were evaluated based on fog onset and dissipation forecast skill, and hourly weather observations recorded between 2011 and 2021 at the airport were analyzed to create a fog climatology of the region for the 11-year period. A short-term persistence analysis of fog states using Markov Chains was also conducted. The results indicate that our models outperformed the current physics-based model, Harmonie-Arome, in several metrics, although fog state transitions remain a weak point. This project combines the learnings from the predictive models and analysis to enhance the fog nowcasting capabilities at Dublin Airport.


## Repo Structure

**final_reports:** Contains the Project Proposal and final Conference Paper in PDF format. Both documents were created using LaTeX.

**images:** All plots and images used in the conference paper are located in this folder. Several of these were left out of the final version, but were useful for analysis.

**notebooks:** Data preparation, analysis, and modelling notebooks are located here. All the modelling notebook names are prefixed with '04.' (04.1_modelling_xgb.ipyb, for instance). **The notebooks were all run on Google Colab unless otherwise specified.**

**results:** Pickle objects containing artefacts from model runs. this was useful for transferring results between notebooks, saving hyperparameter optimisation progress, and avoiding having to retrain models.

**scripts:** Python modules for visualisations and modelling (including unit tests). The visualisation module is built on matplotlib, seaborn, numpy, and pandas. The modelling module is built on scikit-learn, xgboost, matplotlib, numpy, and pandas.


## Links:
Google Drive version of repo [here](https://drive.google.com/drive/folders/1Z8qqk1vNF_jypLlJQ94Dn2vix-UzDCF5?usp=sharing). This was our main working directory throughout the course of the project. 

LaTeX project containing the Proposal and Conference Paper writeup can be found [here](https://www.overleaf.com/read/mcwxprzcqxky). 


The project dashboard is [here](https://projects.computing.dcu.ie/project.html?module=ca4021).
