import sys, os, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

class DBpediaTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: DistilBertTokenizer, max_length: int = 512):
        self.texts = [f"title: {t} [SEP] content: {c}" for t, c in zip(df["title"], df["content"])]
        self.labels = df["label_id"].tolist() if "label_id" in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(self.texts[idx], padding = "max_length", truncation = True, max_length = self.max_length, return_tensors = "pt")
        item = {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype = torch.long)
        return item

def load_model_and_tokenizer() -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels = 14)
    model.load_state_dict(torch.load("best_model.pt", map_location = "cpu"))
    model.eval()
    return model, tokenizer

def evaluate_model(model: DistilBertForSequenceClassification, tokenizer: DistilBertTokenizer) -> None:
    if not os.path.exists("test_set.parquet"):
        raise FileNotFoundError("test_set.parquet not found.")
    df_test = pd.read_parquet("test_set.parquet")

    le = LabelEncoder()
    le.fit(df_test["label"])
    df_test["label_id"] = le.transform(df_test["label"])
    class_names = list(le.classes_)

    dataset = DBpediaTestDataset(df_test, tokenizer)
    loader = DataLoader(dataset, batch_size = 16, shuffle = False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            preds = model(input_ids = ids, attention_mask = mask).logits.argmax(dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average = None, zero_division = 0)

    print(f"\ntest accuracy: {acc:.4f}\n")
    print("{:<25} {:<10} {:<10} {:<10}".format("class", "precision", "recall", "f1"))
    for c, pr, re, f in zip(class_names, p, r, f1):
        print(f"{c:<25} {pr:.4f}     {re:.4f}     {f:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize = (9, 7), dpi = 200)
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues_r", xticklabels = class_names, yticklabels = class_names, cbar = False, linewidths = 0.4, linecolor = "gray")
    plt.title("confusion matrix – dbpedia-14 test set", pad = 12)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.xticks(rotation = 45, ha = "right", fontsize = 9)
    plt.yticks(rotation = 0, fontsize = 9)
    plt.tight_layout()
    plt.savefig("graphs/confusion_matrix.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("\nconfusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    evaluate_model(model, tokenizer)