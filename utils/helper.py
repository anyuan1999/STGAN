from calculate_metrics import calculate_metrics
from Get_Adjacent import Get_Adjacent


def helper(MP, all_pids, GP, edges, mapp):
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    two_hop_gp = Get_Adjacent(GP, mapp, edges, 2)
    two_hop_tp = Get_Adjacent(TP, mapp, edges, 2)
    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    # 更新以包含准确率
    prec, rec, fscore, FPR, TPR, acc = calculate_metrics(TP, FP, FN, TN)
    print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN},True Negatives: {TN}")
    print(f"Precision: {round(prec, 4)}, Recall: {round(rec, 4)}, Fscore: {round(fscore, 4)}")
    print(f"Accuracy: {round(acc, 4)}")  # 输出准确率

    return TPL, FPL