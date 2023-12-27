import torch.nn as nn

class BoxTrackingLoss(nn.Module):
    classification_weight = 1.0
    regression_weight = 10.0
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)
    smooth_L1_loss = nn.SmoothL1Loss(reduction='none')
    def __init__(self):
        super().__init__()

    def forward(self, network_output, item):
        classification_prediction = network_output[:, :2, ...]
        regression_prediction = network_output[:, 2:, ...]

        classification_targets = item["classification_targets"].unsqueeze(0).long()
        regression_targets = item["regression_targets"].unsqueeze(0)

        loss = self.cross_entropy_loss.forward(classification_prediction, classification_targets)
        regression_loss = self.smooth_L1_loss.forward(regression_prediction, regression_targets)
        regression_loss[classification_targets.unsqueeze(1).expand(regression_loss.size()) == 0] = 0
        #regression_loss = regression_loss[:, :6, ...]
        return self.classification_weight*loss + self.regression_weight*regression_loss.mean()