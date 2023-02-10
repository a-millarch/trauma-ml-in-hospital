# Comparing machine learning approaches for Danish Trauma Care
Code accompanying by study: "Assesing optimal methods for transferring machine learning models to low-volumne and imbalanced clinical datasets– Experiences from predicting outcomes for Danish trauma patients"

## How to use
Pre-procesing is found in src/data/preprocessing.py but requires both TQIP and DTR datasets readily available in /data/raw/ to be run.

The training proces is done in two seperate Python scripts: one for tree-based models and one for neural network models.
A demonstration of how to run the trainingscripts with argparse is available in the "1.0-Training.ipynb" notebook. 

