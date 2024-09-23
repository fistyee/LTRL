import torch

def accuracy(output, target, return_length=False):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()

        # ------------------------------------ #
        # 获取总类别数量
        num_classes = output.shape[1]
        per_class_correct = torch.zeros(num_classes).long()
        per_class_total = torch.zeros(num_classes).long()
        # 对每一个类别进行计算
        for cls_idx in range(num_classes):
            mask = (target == cls_idx)
            per_class_correct[cls_idx] = torch.sum(pred[mask] == target[mask]).item()
            per_class_total[cls_idx] = mask.sum().item()

        # 计算每个类别的accuracy
        per_class_accuracy = per_class_correct.float() / (per_class_total.float() + 1e-10)
        # print(per_class_accuracy)

    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)
    
def top_k_acc(output, target, k=5, return_length=False):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)
