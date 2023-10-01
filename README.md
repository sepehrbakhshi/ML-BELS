# ML-BELS
Balancing Efficiency vs. Effectiveness and Providing Missing Label Robustness in Multi-Label Stream Classification

Available works addressing multi-label classification in a data stream environment focus on proposing accurate models; however, these models often exhibit inefficiency and cannot balance effectiveness and efficiency. In this work, we propose a neural network-based approach that tackles this issue and is suitable for high-dimensional multi-label classification. Our model uses a selective concept drift adaptation mechanism that makes it suitable for a non-stationary environment. Additionally, we adapt our model to an environment with missing labels using a simple yet effective imputation strategy and demonstrate that it outperforms a vast majority of the state-of-the-art supervised models. To achieve our purposes, we introduce a weighted binary relevance-based approach named ML-BELS using the Broad Ensemble Learning System (BELS) as its base classifier. Instead of a chain of stacked classifiers, our model employs independent weighted ensembles, with the weights generated by the predictions of a BELS classifier. We show that using the weighting strategy on datasets with low label cardinality negatively impacts the accuracy of the model; with this in mind, we use the label cardinality as a trigger for applying the weights. We present an extensive assessment of our model using 11 state-of-the-art baselines, five synthetics, and 13 real-world datasets, all with different characteristics. Our results demonstrate that the proposed approach ML-BELS is successful in balancing effectiveness and efficiency, and is robust to missing labels and concept drift.

# Datasets
All the real and synthetic datasets are available. 

Google Drive Link: https://drive.google.com/drive/folders/1MGarx6A94uf2BAGESA0KPDBk7CEK7QLZ?usp=sharing

# Requirements
Python: 3.10.9 <br />
Numpy: 1.23.5 <br />
Pandas:  1.5.3 <br />

# Running ML-BELS

To execute the code, ensure that all the code files and the dataset (in .CSV format) are placed within the same folder. In the BELS_test.py file, make sure to include your dataset name using the "dataset_name" variable in the format: dataset_name = "YOUR_DATASET_NAME". After making this change, run the BELS_test.py file.

# Citing BELS

```plaintext

```
