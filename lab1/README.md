# AHLT

This work is based in the Named Entity Recognition and Classification (NERC) task proposed in SemEval 2013, the 7th International Workshop on Semantic Evaluations. In particular, the main goal of this report is to extract relevant Named Entities (NE) from the provided biomedical texts and classifying them into four main groups: “drug”, “drug n”, “group” and “brand”.
On the other hand, the scope of this lab was to explore the data provided in order to understand it better and be able to build two types of classifiers. The first one is a rule-based classifier, which given a tokenized sentence it will classify each token into 5 groups: “drug”, “drug n”, “group”, “brand” and “NONE”.

For example, the following sentence: “Maximal exercise testing, a maneuver often applied to cardiac patients, does not significantly alter the serum digoxin level.” contains the drug digoxin. Consequently, rules will have to be developed so that (ideally) every token gets a prediction of “NONE” except digoxin.

Once the performance of this classifier has been assessed, a set of handcrafted features will be designed in order to feed them to a Machine Learning based system. Finally, the 2 developed systems will be compared to determine the one that has a better performance, according to a selected metric. In the chapters below, the work performed to solve this task will be explained, as well as the results obtaine

## Report
A more detailed analysis of this task is shown in the [report](./report.pdf) of this task.


## Instructions


Steps to run the script

### Running script for the first time
This section shows how to create a virtual environment to run the scripts in this repository
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running the following command (installed dependencies should be displayed)
```bash
pip list
```

5. Close virtual env
```bash
deactivate
```

## Execute scripts

1.open virtual env
```bash
source venv/bin/activate
```
2. Running the script

	2.1.  For **Rule-based NERC** execute
	```bash
    python baseline-NER.py data/devel out.txt
	```

	2.2. For **ML based NERC** execute
	```bash
	./ahlt.sh
	```

3. Close virtual env
```bash
deactivate
```


## Contributors

Valeriu Vicol and Guillermo Creus
