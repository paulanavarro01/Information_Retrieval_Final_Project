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
    irwa-venv/bin/activate
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

## PART 2

### How to run the code? 

In order to run the second part of the project the first part has to be already executed.

Then, in order to execute the code, go to
```bash

/project_progress/part_2
```
Execute first the file **indexing_evaluation.py** to see the completed activity 1.

And after that go to **evaluation_metrix.ipynb*** and execute each cell in order.


## PART 3

### How to run the code? 

In order to run the third part of the project the first part and second part have to be already executed.

Then, in order to execute the code, go to
```bash

/project_progress/part_3
```
First, install in your terminal 

```bash
#pip install gensim
#pip install tabulate
```
and then you can already execute all the code in **rankings.ipynb**

## PART 4

This is the last part of the Information and Web Retrieval project, in which a fully functional search engine has been extended with a complete web interface, a Retrieval-Augmented Generation system, and comprehensive web analytics.

### How to run the code? 

Execute this command:

```bash
python web_app.py
```