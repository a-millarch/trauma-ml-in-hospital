# Comparing machine learning approaches for Danish Trauma Care
Code accompanying by study: "Assessing optimal methods for transferring machine learning models to low-volume and imbalanced clinical datasets: experiences from predicting outcomes of Danish trauma patients" available at: 
https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2023.1249258/full

## How to use
Pre-procesing is found in src/data/preprocessing.py but requires both TQIP and DTR datasets readily available in /data/raw/ to be run. 

TQIP data is available from the American College of Surgeons (ACS), free of charge for members of the TQIP community or for a fee for non-members. Data requests can be made through the TQIP website (https://www.facs.org/quality-programs/trauma/quality/trauma-quality-improvement-program/). 
DTD data is available upon reasonable request for research purposes through the Danish Regions Quality Control Program (RKKP) through their website (https://www.rkkp.dk/forskning/).
Due to confidentiality issues, we are not at liberty to share the Electronic Health Record data used in this study.

The training proces is executed in two seperate Python scripts: one for tree-based models and one for neural network models. The training proces and model performance is tracked using MLflow. 
The study was conducted in an AzureML environment, which is made apparent in requirement.txt. It's possible to use the project files in a local environment with some adaptation particularly to the data pre-processing script, which is highly specific for the Danish dataset and TQIP. 

A demonstration of how to run the training scripts with argparse is available in the "1.0-Training.ipynb" notebook. 

