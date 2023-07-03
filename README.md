# Comparing machine learning approaches for Danish Trauma Care
Code accompanying by study: "Assesing optimal methods for transferring machine learning models to low-volumne and imbalanced clinical datasets– Experiences from predicting outcomes for Danish trauma patients"

## How to use
Pre-procesing is found in src/data/preprocessing.py but requires both TQIP and DTR datasets readily available in /data/raw/ to be run. 

TQIP data is available from the American College of Surgeons (ACS), free of charge for members of the TQIP community or for a fee for non-members. Data requests can be made through the TQIP website (https://www.facs.org/quality-programs/trauma/quality/trauma-quality-improvement-program/). 
DTD data is available upon reasonable request for research purposes through the Danish Regions Quality Control Program (RKKP) through their website (https://www.rkkp.dk/forskning/).
Due to confidentiality issues, we are not at liberty to share the Electronic Health Record data used in this study. 

The training proces is done in two seperate Python scripts: one for tree-based models and one for neural network models.
A demonstration of how to run the trainingscripts with argparse is available in the "1.0-Training.ipynb" notebook. 

