# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

__all__ = ["trend_rcnn_inference", "TrendRCNNOutputLayers"]

#print(__name__)
logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def trend_rcnn_inference(boxes, scores, attributes, image_shapes, score_thresh, nms_thresh, topk_per_image, attr_score_thresh, num_attr_classes, max_attr_pred):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        trend_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, attributes_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, attr_score_thresh, num_attr_classes, max_attr_pred
        )
        for scores_per_image, boxes_per_image, attributes_per_image, image_shape in zip(scores, boxes, attributes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def trend_rcnn_inference_single_image(
    boxes, scores, attributes, image_shape, score_thresh, nms_thresh, topk_per_image, attr_score_thresh, num_attr_classes, max_attr_pred
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        attributes = attributes[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    #print("Printing the number of classes in the box: ", num_bbox_reg_classes)
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    num_attr_reg_classes = attributes.shape[1] // num_attr_classes
    # [ANMOL] this just prints the number of object classes that we have... here its 46
    attributes = attributes.view(-1, num_attr_reg_classes, num_attr_classes)
    # [ANMOL] reshaped the attributes [proposals, objectclass, attrclass]

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # filter mask shape is same as score shape: [proposals, obj classes]
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    # there would be more indices/proposals after this compared as more number of scores might be >
    # greater than threshold would be interesting to check how it would work class agnostic attr classification
    # might fail there.. In the current example: R=1000, but R'=45806
    #print("filter ind shape: ", filter_inds.shape)

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    #before this scores shape was [R,num_classes], after filter mask it will just convert to [R']

    if num_attr_reg_classes == 1:
        attributes = attributes[filter_inds[:, 0], 0]
    else:
        attributes = attributes[filter_mask]
    #BOTH of these should produce attribute of shape [R', attr_classes]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, attributes = boxes[keep], scores[keep], filter_inds[keep], attributes[keep]

    attributes[attributes < attr_score_thresh] = 0
    attr_scores_sorted, attr_indices = torch.sort(attributes, 1, descending=True)
    attr_indices[attr_scores_sorted < attr_score_thresh] = 294
    attributes_inds = attr_indices[:, 0:max_attr_pred]
    #del attr_indices

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.attr_scores = attributes
    result.attr_classes = attributes_inds
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class TrendRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        pred_attr_class_logits,
        proposals,
        num_attr_classes,
        max_attr_pred,
        ignore_nan_attr_class = 0,
        smooth_l1_beta=0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.pred_attr_class_logits = pred_attr_class_logits
        self.num_attr_classes = num_attr_classes
        self.max_attr_pred = max_attr_pred
        self.ignore_nan_attr_class = ignore_nan_attr_class
        self.smooth_l1_beta = smooth_l1_beta
        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                #print("gt_classes: ", self.gt_classes.shape)
                #GT CLasses is just proposals with class Labels: For BS=4, shape = (2048)
                #device = self.pred_proposal_deltas.device
                num_classes = self.pred_class_logits.shape[1] - 1
                #print("num_classes: ", num_classes)
                #self.gt_attr_classes = cat([torch.randint(0,100,(num_classes * 14,), device=self.pred_proposal_deltas.device) for p in proposals], dim=0)
                self.gt_attr_classes = cat([p.gt_attributes.get_tensor() for p in proposals], dim=0)
                #self.gt_attr_classes = self.gt_attr_classes.get_tensor()
                #print(self.gt_attr_classes.get_tensor())
                #print(self.gt_attr_classes.shape)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("trend_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("trend_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("trend_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def binary_cross_entropy_loss(self):
        """
        Compute the cross_entropy_loss for attribute Classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_attr_class_logits.sum()
        #gt_proposal_deltas = self.box2box_transform.get_deltas(
        #    self.proposals.tensor, self.gt_boxes.tensor
        #)
        #ignore_nan_attr_class = 0
        attr_dim = self.max_attr_pred
        #box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_attr_reg = self.pred_attr_class_logits.size(1) == self.num_attr_classes
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # print("BG Class Ind", bg_class_ind)
        # BG Class Ind = 46: Number of Classes - background
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        #print("fg_ind: ", fg_inds)
        # fg_ind: Has indices of Proposal with valid classes
        # For Ex: There are 130 valid proposals. So dimension would be [130]
        if cls_agnostic_attr_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            #print("Entered Loss for Attr Class Agnostic")
            gt_class_cols = torch.arange(self.num_attr_classes, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            #[ANMOL] print("fg_gt_classes: ", fg_gt_classes.shape)
            #[ANMOL] fg_gt_classes: has classes for valid indices of proposals found by fg_inds
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = self.num_attr_classes * fg_gt_classes[:, None] + torch.arange(self.num_attr_classes, device=device)
            # [ANMOL] Dimensions of gt_class_cols [validproposals, 295], This indices will be different values based on object class
            # [ANMOL] Finds cols as we have num_classes * attr_class columns in total in else case.
            # [ANMOL] gt_class_cols has indices of valid columns [validProposals, 295]

        gt_attr_tensor = torch.zeros((self.gt_attr_classes[fg_inds].shape[0], self.num_attr_classes), device=device)
        # [ANMOL] gt_attr_class shape has gt class numbers for all proposals [2048, 14]
        # [ANMOL] gt_attr_classes[fg_inds] select valid proposals from gt: [validproposal, 14]
        # [ANMOL] gt_attr_tensor is just one hot repr for gt_attr_classes: [validproposal, 294]
        gt_attr_tensor.scatter_(1, self.gt_attr_classes[fg_inds], 1)
        # [ANMOL] Set all valid classes to 1 in gt_attr_tensor 1-hot tensor. Confirmed it by sort function.
        #postive_negative_sampling = 14*3
        remove_allzeros_rows = 1
        focal_loss_enabled = 1
        if self.ignore_nan_attr_class:
            #print("Before removing 0s shape Pred: ", self.pred_attr_class_logits[fg_inds[:, None], gt_class_cols][:,:-1].shape)
            #print("Before removing 0s shape gt: ", gt_attr_tensor[:, :-1].shape)
            #print("Before remove 0s gt: ", torch.sort(gt_attr_tensor[:,:-1], 1, descending=True))
            #print("Entered Ignore Nan")
            gt_attr_tensor_t = gt_attr_tensor[:, :-1]
            pred_attr_class_logits_t = self.pred_attr_class_logits[fg_inds[:, None], gt_class_cols][:,:-1]
            #if positive_negative_sampling > 0:
            #find indices of positive samples
            #pass
            #find indices of 14*3 - gts false postive samples for loss
            if remove_allzeros_rows:
                #print("Entered Remove all Zeros")
                pred_attr_class_logits_t = self.pred_attr_class_logits[fg_inds[:, None], gt_class_cols][:,:-1][gt_attr_tensor_t.sum(dim=1) != 0]
                gt_attr_tensor_t = gt_attr_tensor_t[gt_attr_tensor_t.sum(dim=1) != 0]
            if focal_loss_enabled:
                #print("Using Focal Loss")
                # Impl from RetinaNet
                #alpha = 0.25
                #gamma = 2
                #p = pred_attr_class_logits_t.sigmoid()
                #pt = p*gt_attr_tensor_t + (1-p)*(1-gt_attr_tensor_t)         # pt = p if t > 0 else 1-p
                #w = alpha*gt_attr_tensor_t + (1-alpha)*(1-gt_attr_tensor_t)  # w = alpha if t > 0 else 1-alpha
                #w = w * (1-pt).pow(gamma)
                #loss_attr_loss = F.binary_cross_entropy_with_logits(pred_attr_class_logits_t, gt_attr_tensor_t, w.detach(), size_average=False)
                #xt = pred_attr_class_logits_t*(2*gt_attr_tensor_t-1)  # xt = x if t > 0 else -x
                #pt = (2*xt+1).sigmoid()
                #w = alpha*gt_attr_tensor_t + (1-alpha)*(1-gt_attr_tensor_t)
                #focal_loss = -w*pt.log() / 2
                #loss_attr_loss = focal_loss.sum()
                #IMPL 2:
                alpha = 1
                gamma = 2
                BCE_loss = F.binary_cross_entropy_with_logits(pred_attr_class_logits_t, gt_attr_tensor_t, reduce=False)
                pt = torch.exp(-BCE_loss)
                F_loss = alpha * (1-pt)**gamma * BCE_loss
                loss_attr_loss = torch.mean(F_loss)
            #B = A[A.sum(dim=1) != 0]
            #loss_attr_loss = F.binary_cross_entropy_with_logits(self.pred_attr_class_logits[fg_inds[:, None], gt_class_cols][:,:-1], gt_attr_tensor[:, :-1])
            #print("After removing 0s shape Pred: ", pred_attr_class_logits_t.shape)
            #print("After removing 0s shape gt: ", gt_attr_tensor_t.shape)
            #print("After remove 0s gt: ", torch.sort(gt_attr_tensor_t, 1, descending=True))
            else:
                loss_attr_loss = F.binary_cross_entropy_with_logits(pred_attr_class_logits_t, gt_attr_tensor_t)
        else:
            loss_attr_loss = F.binary_cross_entropy_with_logits(self.pred_attr_class_logits[fg_inds[:, None], gt_class_cols], gt_attr_tensor)
        # [ANMOL] predi_attr_class_logits in case of class dependency has dimension [allProposals, num_class*attr_classes]
        # [ANMOL] fg_ind has the valid proposals, for each proposal gt_class_cols as right set of 294 (attr_class) columns indices
        # [ANMOL] pred_attr_class_logits dimension again would be [validProposals, 294]


        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        #loss_attr_loss = loss_attr_loss
        #/ self.gt_classes.numel()
        #loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_attr_loss

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
            "loss_attr_cls": self.binary_cross_entropy_loss(),
        }

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return trend_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class TrendRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
      (3) attributes scores
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        num_attr_classes,
        max_attr_pred,
        attr_cls_mode,
        attr_cls_agnostic,
        ignore_nan_attr_class,
        test_attr_score_thresh = 0.5,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            num_attr_classes (int): number of attributes classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        #print("Is class agnostic: ", attr_cls_agnostic)
        if attr_cls_agnostic:
            num_attr_reg_classes = 1
        else:
            num_attr_reg_classes = num_classes
        if attr_cls_mode == 0:
            self.attr_cls_score = Linear(input_size, num_attr_reg_classes * num_attr_classes)
            nn.init.normal_(self.attr_cls_score.weight, std=0.01)
            nn.init.constant_(self.attr_cls_score.bias, 0)
        elif attr_cls_mode == 1:
            self.attr_cls_score_1 = Linear(input_size, 1024)
            self.attr_cls_score_2 = Linear(1024, num_attr_reg_classes * num_attr_classes)
            nn.init.normal_(self.attr_cls_score_1.weight, std=0.01)
            nn.init.constant_(self.attr_cls_score_1.bias, 0)
            nn.init.normal_(self.attr_cls_score_2.weight, std=0.01)
            nn.init.constant_(self.attr_cls_score_2.bias, 0)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
        self.num_attr_classes = num_attr_classes#295
        self.max_attr_pred = max_attr_pred
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.test_attr_score_thresh = test_attr_score_thresh
        self.attr_cls_mode = attr_cls_mode
        self.attr_cls_agnostic = attr_cls_agnostic
        self.ignore_nan_attr_class = ignore_nan_attr_class

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_attr_classes"      : cfg.MODEL.ROI_HEADS.NUM_ATTR_CLASSES,
            "max_attr_pred"         : cfg.MODEL.ROI_HEADS.MAX_ATTR_PRED,
            "attr_cls_mode"         : cfg.MODEL.ROI_HEADS.ATTR_CLS_MODE,
            "attr_cls_agnostic"     : cfg.MODEL.ROI_HEADS.ATTR_CLS_AGNOSTIC,
            "ignore_nan_attr_class" : cfg.MODEL.ROI_HEADS.IGN_NAN_ATTR_CLS,
            "test_attr_score_thresh": cfg.MODEL.ROI_HEADS.ATTR_SCORE_THRESH_TEST,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE
            # fmt: on
        }

    def forward(self, x):
        """
        Returns:
            Tensor: Nx(K+1) scores for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
            Tensor: Nx(KxAttributes) attr scores for each box.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        if self.attr_cls_mode == 0:
            attr_scores = self.attr_cls_score(x)
        elif self.attr_cls_mode == 1:
            attr_scores = self.attr_cls_score_2(F.relu(self.attr_cls_score_1(x)))
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas, attr_scores

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas, attr_scores = predictions
        return TrendRCNNOutputs(
            self.box2box_transform, scores, proposal_deltas, attr_scores, proposals, self.num_attr_classes, self.max_attr_pred, self.ignore_nan_attr_class, self.smooth_l1_beta
        ).losses()

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        attributes = self.predict_attributes(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return trend_rcnn_inference(
            boxes,
            scores,
            attributes,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.test_attr_score_thresh,
            self.num_attr_classes,
            self.max_attr_pred
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


    def predict_attributes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        _, _, attr_scores = predictions
        num_inst_per_image = [len(p) for p in proposals]
        #print(attr_scores.shape)
        probs = F.sigmoid(attr_scores)
        #print(probs.shape)
        return probs.split(num_inst_per_image, dim=0)
