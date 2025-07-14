import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve

# 1. Load Data
probs = pd.read_csv('predicted_proba.csv').values.flatten().tolist()
true_labels = pd.read_csv('true_labels.csv').values.flatten().tolist()

# 2. Custom Functions

def predict(pred_probs, threshold):
    """
    Takes a list of prediction probabilities and a threshold value.
    Returns a list of predicted class labels (0 or 1).
    """
    return [1 if prob >= threshold else 0 for prob in pred_probs]

def acc_score(true_labels, preds):
    """
    Calculates the accuracy score.
    """
    correct = sum([1 for true, pred in zip(true_labels, preds) if true == pred])
    return correct / len(true_labels)

def error_rate(true_labels, preds):
    """
    Calculates the model error rate using the accuracy score.
    """
    return 1 - acc_score(true_labels, preds)

def prec_recall_score(true_labels, preds):
    """
    Calculates model precision and recall.
    Returns (precision, recall).
    """
    TP = FP = FN = 0
    for true, pred in zip(true_labels, preds):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall

def f_beta(true_labels, preds, beta):
    """
    Computes the F_beta score for any beta value.
    """
    precision, recall = prec_recall_score(true_labels, preds)
    beta_sq = beta ** 2
    if (beta_sq * precision + recall) == 0:
        return 0.0
    f_beta_score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return f_beta_score

def TPR_FPR_score(true_labels, preds):
    """
    Computes True Positive Rate (TPR/Recall) and False Positive Rate (FPR).
    Returns (TPR, FPR).
    """
    TP = FP = FN = TN = 0
    for true, pred in zip(true_labels, preds):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN += 1
        elif pred == 0 and true == 0:
            TN += 1
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return TPR, FPR

def roc_curve_computer(true_labels, pred_probs, thresholds):
    """
    Computes lists of TPR and FPR values for each threshold in thresholds.
    Returns: (list_of_TPR, list_of_FPR)
    """
    TPR_list = []
    FPR_list = []
    for thresh in thresholds:
        preds = predict(pred_probs, thresh)
        TPR, FPR = TPR_FPR_score(true_labels, preds)
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    return TPR_list, FPR_list

# 3. Example usage

thresh = 0.5
preds = predict(probs, thresh)

print("Model Predictions:", preds)

accuracy = acc_score(true_labels, preds)
print("Model Accuracy:", accuracy)
print("Sklearn Accuracy:", accuracy_score(true_labels, preds))

error = error_rate(true_labels, preds)
print("Model Error Rate:", error)

precision, recall = prec_recall_score(true_labels, preds)
print("Precision =", precision)
print("Recall =", recall)
print("Sklearn Precision Score =", precision_score(true_labels, preds))
print("Sklearn Recall Score =", recall_score(true_labels, preds))

F1 = f_beta(true_labels, preds, 1)
print("F1 =", F1)
print("Sklearn F1 Score =", f1_score(true_labels, preds))

TPR, FPR = TPR_FPR_score(true_labels, preds)
print("TPR =", TPR)
print("FPR =", FPR)

# ROC Curve
thresholds = [i / 100 for i in range(0, 101)]
TPR_list, FPR_list = roc_curve_computer(true_labels, probs, thresholds)

# Sklearn ROC Curve for comparison
fpr, tpr, thresholds_sklearn = roc_curve(true_labels, probs)

 Plotting (optional, uncomment if needed)
import matplotlib.pyplot as plt
plt.plot(FPR_list, TPR_list, label="Custom ROC")
plt.plot(fpr, tpr, linestyle='--', label="sklearn ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
