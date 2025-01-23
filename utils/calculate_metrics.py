def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
    acc = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0  # 新增准确率计算

    return prec, rec, fscore, FPR, TPR, acc