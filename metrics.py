import torch
import torch.nn as nn


""" Calculation of matrix distance as in scikit-learn but on rows. 
    Do the average for row (and so for year). """
class Rec_Loss(nn.Module):
    """ Initialize configration. """
    def __init__(self):
        super(Rec_Loss, self).__init__()
    
    """ Method used to compute the loss value. """
    def forward(self, y_true, y_pred):
        out = torch.sub(y_true, y_pred).pow(2)
        # # dim = 2 indicates to pick columns 
        # # element per row to compute the mean per row
        out = torch.mean(input=out, dim=2)  # , keepdim=True)
        out = torch.sqrt(out)
        out = torch.mean(out)

        return out





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
