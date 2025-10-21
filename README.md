# Information Retrieval and Web Analytics (IRWA) - Final Project template

# PART 1

In the Part1 folder you can find the "data_processing.py" file, this file has been created in order to complete the first part of the project: Data preparation and Index Creation.
This prepares the dataset for the following stages of the Information Retrieval Project.

There are some files involved in this first part of the project:
- *data/fashion_products_dataset.json*: this is the original dataset without preprocessing, it has been used to understand what we are working with and how to do it.
- *part_1/data_preprocessing.py*: this python file is used in order to preprocess the data in the json file, here 4 functions have been defined in order to prepare the data for the following stages of the project.
- *data/processed_docs.jsonl*: this is the output file with the cleaned fields for each document.
- *data/inverted_index.json*: this document is the inverted index of our original dataset.

In order to execute the code the only thing that has to be done is executing the file *part_1/data_preprocessing.py*, this will create the new files *data/processed_docs.jsonl* and *data/inverted_index.json*.





