# Information Retrieval and Web Analytics (IRWA) - Final Project

This repository contains the final project implementation for the course **Information Retrieval and Web Analytics (IRWA)**. The project focuses on building a basic Information Retrieval system based on a fashion product dataset, covering the stages of data preprocessing, indexing, ranking, and evaluation.

## PART 1

### How to run the code? 

In order to execute the code these are the steps to follow:
#### 1. Environment Setup and Installation

All project code must be executed from within the dedicated virtual environment named `irwa-venv`.
1.  **Create the Virtual Environment:**
    ```bash
    python3 -m venv irwa-venv
    ```

2.  **Activate the Environment:**
    ```bash
    source irwa-venv/bin/activate
    ```

3.  **Install Dependencies:**
    With the environment active, install the necessary libraries. 
    ```bash
    pip install -r requirements.txt
    ```

#### 2. Dataset Preprocess and Cleaning

Execute the preprocessing script. This file is responsible for cleaning the dataset, applying tokenization, and creating the inverted index.

```bash

python part_1/data_preprocessing.py
```
After this script is executed, two files will be created inside the data folder.

```bash

processed_docs.jsonl
```
Which is the cleaned and preprocessed datased. And
```bash

inverted_index.json
```
Which is the file with the inverted index.

#### 3. Exploratory Data Analyis


For the second activity, a Jupiter notebook has been created inside the part_1 folder: 
```bash

exploratory_data_analysis.ipynb
```
In order to execute this file it's important that the two previous parts have already been executed, since in the file the cleaned dataset **processed_docs.jsonl** will be analyzed.

In order to see all the outputs all that has to be done is executing the cells in order.


