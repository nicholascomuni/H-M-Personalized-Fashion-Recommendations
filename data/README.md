# Data

The data is **not included** in this repository. It belongs to the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
competition and is subject to its rules. Accept the rules on the competition page before downloading.

## Download with the Kaggle CLI

The full archive includes the product images, which the notebook does not use. Download only the CSV files:

```bash
pip install kaggle   # needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)
COMP=h-and-m-personalized-fashion-recommendations
for f in articles.csv customers.csv transactions_train.csv sample_submission.csv; do
  kaggle competitions download -c "$COMP" -f "$f" -p data/
done
cd data && for z in *.zip; do unzip -o -q "$z" && rm "$z"; done   # the CLI may deliver files zipped
```

## Expected layout

```text
data/
├── articles.csv
├── customers.csv
├── transactions_train.csv      # ~31.8M rows
├── sample_submission.csv
└── reduced/                    # created by the notebook (compact parquet copies)
    ├── articles.parquet.gzip
    ├── customers.parquet.gzip
    └── transactions.parquet.gzip
```

In a Kaggle notebook the competition files are mounted at `../input/h-and-m-personalized-fashion-recommendations`:
set `DATA_DIR` to that path and `REDUCED_DIR` to a writable folder such as `/kaggle/working/reduced`.
