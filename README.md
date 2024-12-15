
## Introduction
This code is for the paper "NeiKG: Knowledge Graph-driven Neighbor Selection and Aggregation for Long-tail Recommendation" submitted to AAAI2025.

## Environment Requirement
```
This code requires deployment on Python 3.7.10, and requires packages as follows:

torch == 1.8.0
numpy == 1.23.5
pandas == 1.5.3
scipy == 1.10.1
tqdm == 4.65.0
scikit-learn == 1.2.2
```

## The detailed steps of NeiKG:
1. Carry out the file doCooccur.py to obtain co-occurrence neighbors
```
python doCooccur.py
```
2. Carry out the file selector.py to obtain long-tail neighbors
```
python selector.py
```
3. Carry out the file main_NeiKG.py to obtain the final recommendation
```
python main_NeiKG.py
```


## Datasets
We provided three datasets to validate NeiKG: Last-FM, MovieLens-1M, and Amazon-book.

|                | Last-FM |MovieLens-1M| Amazon-book |
| :------------: | :-----: |  :-----:   |:-----:   |
|    n_users     |  23566  |    6040    | 70679 |
|    n_items     |  48123  |    3655    |24915|
| n_interactions | 3034796 |   997579   |847733|


We mapped items of three datasets to Freebase entities to construct KG.
The following table shows the KG information of three datasets:

| Knowledge Graph |   Freebase(Last-FM)   |  Freebase(MovieLens-1M)  | Freebase(Amazon-book)
|:---------------:|          :-----------:         |     :-------:     |:-------:     |
|   #entities    |              58266            |       64731     |88572|
|   #relations   |                 9              |        9        |39|
|    #triples    |              464567            |      41688     |2557746|

