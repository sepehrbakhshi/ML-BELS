# ML-BELS
Balancing Efficiency vs. Effectiveness and Providing Missing Label Robustness in Multi-Label Stream Classification

Available works addressing multi-label classification in a data stream environment focus on proposing accurate prediction models; however, they struggle to balance effectiveness and efficiency. In this work, we present a neural network-based approach that tackles this issue and is suitable for high- dimensional multi-label classification. The proposed model uses a selective concept drift adaptation mechanism that makes it well-suited for a non-stationary environment. We adapt the model to an environment with missing labels using a simple imputation strategy and demonstrate that it outperforms a vast majority of the supervised models. To achieve these, a weighted binary relevance-based approach named ML-BELS is introduced. To capture label dependencies, instead of a chain of stacked classifiers, the proposed model employs independent weighted ensembles as binary classifiers, with the weights generated by the predictions of a BELS classifier. We present an extensive assessment of the proposed model using 11 prominent baselines, five synthetic, and 13 real-world datasets, all with different characteristics. The results demonstrate that the proposed approach ML-BELS is successful in balancing effectiveness and efficiency, and is robust to missing labels and concept drift.

# Datasets
All the real and synthetic datasets are available. 

Google Drive Link: https://drive.google.com/drive/folders/1MGarx6A94uf2BAGESA0KPDBk7CEK7QLZ?usp=sharing

# Requirements
Python: 3.10.9 <br />
Numpy: 1.23.5 <br />
Pandas:  1.5.3 <br />

# Running ML-BELS

To execute the code, ensure that all the code files and the dataset (in .CSV format) are placed within the same folder. In the Main_missing_labels.py file, make sure to include the following:

1- Your dataset name using the "dataset_name" variable in the format: dataset_name = "YOUR_DATASET_NAME"; <br />
2- Variable named missing_percentage indicates the missing label perentage, and should be a float between 0 and 1. Zero means that there are no missing labels and one means all labels are missing;<br />
3- Variable named label_count_full indicates the number of labels in the dataset. Make sure to enter the number of labels for each dataset before running the code;<br />
4- You can modify the chunk size by changing the chunk_size variable to your desired values.<br />

 After making these changes, run the Main_missing_labels.py file.

# Citing ML-BELS

```plaintext
@article{BAKHSHI2024111489,
title = {Balancing efficiency vs. effectiveness and providing missing label robustness in multi-label stream classification},
journal = {Knowledge-Based Systems},
pages = {111489},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.111489},
url = {https://www.sciencedirect.com/science/article/pii/S0950705124001242},
author = {Sepehr Bakhshi and Fazli Can},
keywords = {Multi-label classification, Data streams, Neural networks, Concept drift, Missing labels},
abstract = {Available works addressing multi-label classification in a data stream environment focus on proposing accurate prediction models; however, they struggle to balance effectiveness and efficiency. In this work, we present a neural network-based approach that tackles this issue and is suitable for high-dimensional multi-label classification. The proposed model uses a selective concept drift adaptation mechanism that makes it well-suited for a non-stationary environment. We adapt the model to an environment with missing labels using a simple imputation strategy and demonstrate that it outperforms a vast majority of the supervised models. To achieve these, a weighted binary relevance-based approach named ML-BELS is introduced. To capture label dependencies, instead of a chain of stacked classifiers, the proposed model employs independent weighted ensembles as binary classifiers, with the weights generated by the predictions of a BELS classifier. We present an extensive assessment of the proposed model using 11 prominent baselines, five synthetic, and 13 real-world datasets, all with different characteristics. The results demonstrate that the proposed approach ML-BELS is successful in balancing effectiveness and efficiency, and is robust to missing labels and concept drift.}
}
```
