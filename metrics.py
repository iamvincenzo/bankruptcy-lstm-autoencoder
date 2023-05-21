import torch
import torch.nn as nn


""" Calculation of matrix distance as in scikit-learn but on rows. 
    Do the average for row (and so for year). """
class Rec_Loss(nn.Module):
    """ Initialize configration. """
    def __init__(self):
        super(Rec_Loss, self).__init__()
    
    """ Method used to compute the loss value. """
    def forward(self, y_pred, y_true):
        out = torch.sub(y_pred, y_true).pow(2)
        # dim = 2 indicates to pick columns element per each row to compute the mean per row
        # esample [[2, 2, 2], [1, 1, 1]] --> [mean-row1=[2+2+2/3], mean-row2=[1+1+1/3]]
        out = torch.mean(input=out, dim=2)
        out = torch.sqrt(out)
        out = torch.mean(out)

        return out

""" Method used to compute the y_true for dense90. """
def reconstruction_for_prior(y_pred, y_true):
    out = torch.sub(y_pred, y_true).pow(2)
    # dim = 2 indicates to pick columns element per each row to compute the mean per row
    # esample [[2, 2, 2], [1, 1, 1]] --> [mean-row1=[2+2+2/3], mean-row2=[1+1+1/3]]
    out = torch.mean(input=out, dim=2)
    out = torch.sqrt(out)

    return out


# https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

""" Method used to compute accuracy, precision and recall. """
def compute_metrics(predictions, targets):
    # returns the indices of the maximum value of all elements in the input tensor (dim=1) by row
    predicted_classes = torch.argmax(predictions, dim=1)

    # correct_predictions = (predicted_classes == targets).sum().item()
    tp = torch.sum((predicted_classes == 1) & (targets == 1)).item()
    fp = torch.sum((predicted_classes == 1) & (targets == 0)).item()
    fn = torch.sum((predicted_classes == 0) & (targets == 1)).item()
    tn = torch.sum((predicted_classes == 0) & (targets == 0)).item()

    # accuracy = (correct_predictions / targets.size(0))
    # 1e-10 is a small value commonly used to avoid division by zero
    accuracy = (tp + tn) / (tp + fp + fn + tn) # + 1e-10)        
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    specificity = tn / (tn + fp)
    
    # tp, fp, tn, fn = confusion(predicted_classes, targets)
    # print(tp, fp, fn, tn)

    # create the confusion matrix
    confusion_matrix = torch.tensor([[tp, fp], 
                                     [fn, tn]])

    return accuracy, precision, recall, f1_score, specificity, confusion_matrix



# def reconstruction_loss(y_true, y_pred):
#     out = torch.sub(y_true, y_pred).pow(2)
#     print("\n\n", out.shape)
#     print("\n\n", out)
#     # dim = 2 indicates to pick columns 
#     # element per row to compute the mean per row
#     out = torch.mean(input=out, dim=2)  # , keepdim=True)
#     print("\n\n", out.shape)
#     print("\n\n", out)
#     out = torch.sqrt(out)
#     print("\n\n", out.shape)
#     print("\n\n", out)
#     out = torch.mean(out)
#     print("\n\n", out.shape)
#     print("\n\n", out)

#     return out

""" Test. 
if __name__ == "__main__":
    x1 = torch.rand((2, 2, 2))
    x2 = torch.rand((2, 2, 2))

    x2[0, 0, 0] = 2

    print("\n", x1, "\n\n", x2, "\n")

    reconstruction_loss(x1, x2)

    criterion = Rec_Loss()

    out = criterion(x1, x2)

    print
"""



# # output-example
# y_true:
# tensor([[[0.6539, 0.2613],
#          [0.3045, 0.5636]],

#         [[0.1454, 0.4544],
#          [0.5622, 0.4754]]])

# y_pred:
# tensor([[[2.0000, 0.3301],
#          [0.4366, 0.1001]],

#         [[0.6967, 0.8455],
#          [0.2044, 0.7300]]])

# y_true-y_pred:
# torch.Size([2, 2, 2])
# tensor([[[1.8119, 0.0047],
#          [0.0175, 0.2148]],

#         [[0.3039, 0.1530],
#          [0.1280, 0.0649]]])

# mean-per-row(y_true-y_pred):
# torch.Size([2, 2])
# tensor([[0.9083, 0.1161],
#         [0.2285, 0.0964]])

# sqrt:
# torch.Size([2, 2])
# tensor([[0.9530, 0.3408],
#         [0.4780, 0.3105]])

# mean-tot (also bacth)
# torch.Size([])
# tensor(0.5206)
