[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/peCDTFbC)

# Block B: Logistic Regression from Scratch

Year 1, Block B (FAI1, 2024-25). My work for the machine learning block. The main goal is to build logistic regression from scratch with numpy (cost function, gradient descent, prediction, evaluation), with and without regularization, and then compare it to scikit-learn. There is also a creative brief notebook with SQL queries, EDA, data processing, and a few models.

## Dataset

The project uses the **Diabetes 130-US Hospitals (1999-2008)** dataset from the UCI Machine Learning Repository.

- UCI page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- File used in the notebooks: `diabetic_data.csv`

The dataset is not in the repo. Download `diabetic_data.csv` from UCI and put it in the `data/` folder so the path `./data/diabetic_data.csv` works. The Week 6 math notebook also uses a small `house-prices.csv`, put it in `data/` too as `./data/house-prices.csv`.

## How to run

```
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook you want.

## Main notebooks

- `LogisticRegression_temp( MOHAMMADALI JABERI )final.ipynb` - logistic regression from scratch, the clean copy to look at first.
- `BlockY1B_CreativeBrief (Mohammadali Jaberi 244437 final 14).ipynb` - the creative brief, latest and clean copy.

## Note on duplicates

This repo keeps the weekly progress copies of the notebooks (first week, second week, third week, final 10, etc.) so the history is visible. They are older snapshots. The clean copies to use are the two listed above. `final 14` is newer than `final 13` (they only differ in the last model cells), and the copy under `Deliverable Block B/` is the same as the root `final 14`.
