Requirements

Pandas: 1.3.5
Python: 3.7.7 (default, Mar 23 2020, 17:31:31) 
[Clang 4.0.1 (tags/RELEASE_401/final)]
NLTK: 3.6.5
Scikit-Learn: 1.0.2
Tensorflow: 2.0.0
Numpy: 1.18.5

Glove-vectors
glove.42B.300d.txt

There are 3 separate files, one each to deal with one dataset.
FNC_1.py corresponds to fnc-1 dataset (4-class)
FNC.py/ FNC.ipynb deals with the FNC data set (2-class)
NELA.py/NELA.ipynb deals with the NELA data set (2-class

The code assumes that fnc-1 (4-class) datasets are in folder named 'FNC_1_Data' 
The code assumes that fnc (2-class) datasets are in folder named 'FNC_Data'
The code assumes that NELA datasets are in folder named 'NELA_Data'
The code assumes glove.42B.300d.txt is present in the same folder

Running each of the file will perform the assignment task (evaluating classifiers: with raw features, with feature selection,
and with dimension reduction) on a particular dataset. The confusion matrices, along with accuracy scores and 
comparison charts are produced upon execution.
Word embedding has been done only on the FNC-1 data set on account of computational bottlenecks.


A separate observations.pdf file with plots and tables has been attached.



