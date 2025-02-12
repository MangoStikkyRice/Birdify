from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def compute_metrics(all_labels, all_preds):
    """Compute common classification metrics."""
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    return accuracy, f1, precision, recall, report
