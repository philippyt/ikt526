## How to Run

### Requirements

Install the required packages:

    pip install -r requirements.txt

### Check files
Check that all of these are in same directory:

- evaluate.py  
- best_model.pt  
- test_set.parquet

### Run

    python evaluate.py

### Output

- Evaluation metrics printed to the terminal  
- Confusion matrix saved to `confusion_matrix.png`
