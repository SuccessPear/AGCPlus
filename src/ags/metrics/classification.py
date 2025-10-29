# src/metrics/classification.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def accuracy(pred: torch.Tensor, target: torch.Tensor, topk=(1,), logits: bool = True):
    """
    Top-k accuracy cho multi-class.
    - pred: (B, C) logits hoặc probs
    - target: (B,) long
    - topk: ví dụ (1,) hoặc (1, 5)
    - logits: True nếu pred là logits
    Trả về dict: {"acc@1": ..., "acc@5": ...}
    """
    if logits:
        pred = F.softmax(pred, dim=1)

    maxk = max(topk)
    # (B, maxk) các vị trí class dự đoán tốt nhất
    _, idx = pred.topk(maxk, dim=1, largest=True, sorted=True)
    idx = idx.t()  # (maxk, B)
    correct = idx.eq(target.view(1, -1).expand_as(idx))  # (maxk, B)

    res = {}
    B = target.size(0)
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        res[f"acc_{k}"] = correct_k / B
    return res


@torch.no_grad()
def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    average: str = "macro",      # "macro" | "micro" | "weighted" | "none"
    num_classes: int | None = None,
    logits: bool = True,
    eps: float = 1e-12,
):
    """
    F1 cho multi-class, không dùng sklearn.
    - pred: (B, C) logits/probs
    - target: (B,) long
    - average:
        * "macro": trung bình đều trên lớp
        * "micro": gộp toàn bộ TP/FP/FN rồi tính
        * "weighted": trung bình theo support (số mẫu từng lớp)
        * "none": trả dict theo từng lớp {"f1_c0": ..., ...}
    - num_classes: mặc định lấy từ pred.shape[1]
    - logits: True nếu pred là logits
    """
    if logits:
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred.argmax(dim=1)

    if num_classes is None:
        num_classes = int(pred.shape[1])

    # Confusion components per-class
    f1_per_class = []
    support = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (target == c)).sum().item()
        fp = ((pred_labels == c) & (target != c)).sum().item()
        fn = ((pred_labels != c) & (target == c)).sum().item()
        sup = (target == c).sum().item()

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1c       = 2 * precision * recall / (precision + recall + eps)

        f1_per_class.append(f1c)
        support.append(sup)

    f1_per_class = torch.tensor(f1_per_class, dtype=torch.float32)
    support = torch.tensor(support, dtype=torch.float32)
    support_sum = support.sum().clamp(min=eps)

    if average == "none":
        return {f"f1_c{c}": float(f1_per_class[c]) for c in range(num_classes)}
    if average == "macro":
        return {"f1_macro": float(f1_per_class.mean().item())}
    if average == "weighted":
        w = support / support_sum
        return {"f1_weighted": float((w * f1_per_class).sum().item())}
    if average == "micro":
        # micro-F1 = micro-precision = micro-recall = TP_all / (TP_all + 0.5*(FP_all+FN_all)) in F1 form
        # Cách chuẩn: tính TP, FP, FN gộp:
        tp_all = sum(((pred_labels == c) & (target == c)).sum().item() for c in range(num_classes))
        fp_all = sum(((pred_labels == c) & (target != c)).sum().item() for c in range(num_classes))
        fn_all = sum(((pred_labels != c) & (target == c)).sum().item() for c in range(num_classes))
        prec = tp_all / (tp_all + fp_all + eps)
        rec  = tp_all / (tp_all + fn_all + eps)
        f1m  = 2 * prec * rec / (prec + rec + eps)
        return {"f1_micro": float(f1m)}
    raise ValueError(f"Unknown average='{average}'")
