"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .semantic_query_alignment import (
    build_all_query_best_iou_target,
    build_query_semantic_supervision,
    build_union_defect_target,
    compose_dual_quality_scores,
    final_score_ranking_loss,
    select_final_score_background,
)
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        loc_near_iou_threshold=0.10,
        loc_near_weight=1.0,
        loc_far_weight=0.10,
        loc_matched_extra_weight=1.0,
        defect_bce_weight=1.0,
        defect_dice_weight=1.0,
        semantic_query_pos_weight=1.0,
        semantic_query_bg_weight=0.25,
        semantic_query_bg_iou_threshold=0.10,
        use_final_bg_rank=False,
        final_bg_topk=5,
        final_bg_iou_threshold=0.10,
        final_bg_start_epoch=10,
        final_bg_margin=0.05,
        final_bg_temperature=0.10,
        loc_quality_power=0.25,
        semantic_quality_power=0.25,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.loc_near_iou_threshold = float(loc_near_iou_threshold)
        self.loc_near_weight = float(loc_near_weight)
        self.loc_far_weight = float(loc_far_weight)
        self.loc_matched_extra_weight = float(loc_matched_extra_weight)
        self.defect_bce_weight = float(defect_bce_weight)
        self.defect_dice_weight = float(defect_dice_weight)
        self.semantic_query_pos_weight = float(semantic_query_pos_weight)
        self.semantic_query_bg_weight = float(semantic_query_bg_weight)
        self.semantic_query_bg_iou_threshold = float(semantic_query_bg_iou_threshold)
        self.use_final_bg_rank = bool(use_final_bg_rank)
        self.final_bg_topk = int(final_bg_topk)
        self.final_bg_iou_threshold = float(final_bg_iou_threshold)
        self.final_bg_start_epoch = int(final_bg_start_epoch)
        self.final_bg_margin = float(final_bg_margin)
        self.final_bg_temperature = float(final_bg_temperature)
        self.loc_quality_power = float(loc_quality_power)
        self.semantic_quality_power = float(semantic_quality_power)
        self.current_epoch = 0
        self.total_epochs = None
        self._alignment_cache = {}
        self._diagnostics = {}

    def set_epoch(self, epoch: int, total_epochs=None):
        self.current_epoch = int(epoch)
        self.total_epochs = None if total_epochs is None else int(total_epochs)

    def _branch_losses(self, branch: str, branch_outputs):
        main_only = {'loc_quality', 'defect', 'query_semantic', 'final_bg_rank'}
        return [loss for loss in self.losses if branch == 'main' or loss not in main_only]

    def _best_iou_target(self, outputs, targets):
        cache_key = int(outputs['pred_boxes'].data_ptr())
        if cache_key not in self._alignment_cache:
            self._alignment_cache[cache_key] = build_all_query_best_iou_target(outputs['pred_boxes'], targets)
        return self._alignment_cache[cache_key]

    @staticmethod
    def _masked_mean(values, mask, zero):
        return values[mask].mean() if mask.any() else zero

    @staticmethod
    def _batch_spearman(predictions, targets):
        correlations = []
        for prediction, target in zip(predictions.detach().float(), targets.detach().float()):
            if prediction.numel() < 2 or prediction.std(unbiased=False) == 0 or target.std(unbiased=False) == 0:
                continue
            pred_rank = prediction.argsort().argsort().float()
            target_rank = target.argsort().argsort().float()
            pred_rank = pred_rank - pred_rank.mean()
            target_rank = target_rank - target_rank.mean()
            denominator = pred_rank.square().sum().sqrt() * target_rank.square().sum().sqrt()
            correlations.append((pred_rank * target_rank).sum() / denominator.clamp_min(1e-12))
        if correlations:
            return torch.stack(correlations).mean().to(predictions.device)
        return predictions.sum() * 0.0

    def loss_loc_quality(self, outputs, targets, indices, num_boxes):
        logits_tensor = outputs.get('pred_loc_quality')
        if logits_tensor is None:
            return {'loss_loc_quality': outputs['pred_logits'].sum() * 0.0}
        logits = logits_tensor.squeeze(-1)
        best_iou = self._best_iou_target(outputs, targets).to(logits.device)
        near_mask = best_iou >= self.loc_near_iou_threshold
        far_mask = ~near_mask
        matched_mask = torch.zeros_like(near_mask)
        for batch_index, (source_indices, _) in enumerate(indices):
            matched_mask[batch_index, source_indices] = True

        bce = F.binary_cross_entropy_with_logits(logits.float(), best_iou.float(), reduction='none')
        zero = logits.sum() * 0.0
        near_loss = self._masked_mean(bce, near_mask, zero)
        far_loss = self._masked_mean(bce, far_mask, zero)
        matched_loss = self._masked_mean(bce, matched_mask, zero)
        loss = (
            self.loc_near_weight * near_loss
            + self.loc_far_weight * far_loss
            + self.loc_matched_extra_weight * matched_loss
        )

        prediction = logits.sigmoid()
        self._diagnostics.update({
            'metric_best_iou_mean': best_iou.mean() if best_iou.numel() else zero,
            'metric_locq_target_mean': best_iou.mean() if best_iou.numel() else zero,
            'metric_locq_pred_mean': prediction.mean() if prediction.numel() else zero,
            'metric_locq_pred_near': self._masked_mean(prediction, near_mask, zero),
            'metric_locq_pred_far': self._masked_mean(prediction, far_mask, zero),
            'metric_locq_pred_matched': self._masked_mean(prediction, matched_mask, zero),
            'metric_locq_best_iou_spearman_batch': self._batch_spearman(prediction, best_iou),
            'metric_locq_target_ge_05_ratio': (best_iou >= 0.5).float().mean() if best_iou.numel() else zero,
        })
        return {'loss_loc_quality': loss}

    def loss_defect(self, outputs, targets, indices, num_boxes):
        logits = outputs.get('pred_defect_logits')
        if logits is None:
            return {'loss_defect': outputs['pred_logits'].sum() * 0.0}
        target = build_union_defect_target(targets, logits.shape[-2:], logits.device, logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = logits.sigmoid().float()
        target_float = target.float()
        intersection = (probability * target_float).flatten(1).sum(-1)
        denominator = probability.flatten(1).sum(-1) + target_float.flatten(1).sum(-1)
        dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        prediction_binary = probability >= 0.5
        target_binary = target_float >= 0.5
        true_positive = (prediction_binary & target_binary).sum().float()
        predicted_positive = prediction_binary.sum().float()
        target_positive = target_binary.sum().float()
        valid_count = sum(int(target_item.get('mask_valid', torch.empty(0)).numel()) for target_item in targets)
        valid_sum = sum(target_item.get('mask_valid', torch.empty(0, device=logits.device)).float().sum() for target_item in targets)
        zero = logits.sum() * 0.0
        self._diagnostics.update({
            'metric_valid_mask_ratio': valid_sum / max(valid_count, 1) if valid_count else zero,
            'metric_defect_dice': 1.0 - dice.detach(),
            'metric_defect_pixel_precision': true_positive / predicted_positive.clamp_min(1.0),
            'metric_defect_pixel_recall': true_positive / target_positive.clamp_min(1.0),
        })
        return {'loss_defect': self.defect_bce_weight * bce + self.defect_dice_weight * dice}

    def loss_query_semantic(self, outputs, targets, indices, num_boxes):
        semantic_quality = outputs.get('pred_sem_quality')
        if semantic_quality is None:
            return {'loss_query_semantic': outputs['pred_logits'].sum() * 0.0}
        semantic_quality = semantic_quality.squeeze(-1).float().clamp(1e-6, 1.0 - 1e-6)
        best_iou = self._best_iou_target(outputs, targets)
        semantic_target, _, positive_mask, far_background_mask = build_query_semantic_supervision(
            best_iou, indices, self.semantic_query_bg_iou_threshold
        )
        # q_sem is a pooled probability rather than a single logit, so use the
        # explicit stable BCE expression (torch BCE on probabilities rejects AMP).
        semantic_target = semantic_target.float()
        bce = -(
            semantic_target * semantic_quality.log()
            + (1.0 - semantic_target) * (1.0 - semantic_quality).log()
        )
        zero = semantic_quality.sum() * 0.0
        positive_loss = self._masked_mean(bce, positive_mask, zero)
        background_loss = self._masked_mean(bce, far_background_mask, zero)
        loss = (
            self.semantic_query_pos_weight * positive_loss
            + self.semantic_query_bg_weight * background_loss
        )
        positive_mean = self._masked_mean(semantic_quality, positive_mask, zero)
        background_mean = self._masked_mean(semantic_quality, far_background_mask, zero)
        self._diagnostics.update({
            'metric_semq_pred_pos': positive_mean,
            'metric_semq_pred_far_bg': background_mean,
            'metric_semq_pos_bg_gap': positive_mean - background_mean,
        })
        return {'loss_query_semantic': loss}

    def loss_final_bg_rank(self, outputs, targets, indices, num_boxes):
        zero = outputs['pred_logits'].sum() * 0.0
        if not self.use_final_bg_rank or self.current_epoch < self.final_bg_start_epoch:
            self._diagnostics.update({
                'metric_final_bg_selected_per_image': zero,
                'metric_final_bg_score_mean': zero,
                'metric_matched_final_score_mean': zero,
                'metric_rank_violation_ratio': zero,
            })
            return {'loss_final_bg_rank': zero}

        final_scores = compose_dual_quality_scores(
            outputs['pred_logits'].sigmoid(),
            pred_loc_quality=outputs.get('pred_loc_quality'),
            pred_sem_quality=outputs.get('pred_sem_quality'),
            loc_quality_power=self.loc_quality_power,
            semantic_quality_power=self.semantic_quality_power,
        )
        best_iou = self._best_iou_target(outputs, targets)
        selected, counts = select_final_score_background(
            final_scores,
            best_iou,
            indices,
            topk=self.final_bg_topk,
            iou_threshold=self.final_bg_iou_threshold,
        )
        loss = final_score_ranking_loss(
            final_scores,
            targets,
            indices,
            selected,
            margin=self.final_bg_margin,
            temperature=self.final_bg_temperature,
        )

        background_values = final_scores[selected].amax(-1) if selected.any() else final_scores.new_zeros((0,))
        positive_parts = []
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            if source_indices.numel():
                labels = targets[batch_index]['labels'][target_indices]
                positive_parts.append(final_scores[batch_index, source_indices, labels])
        positive_values = torch.cat(positive_parts) if positive_parts else final_scores.new_zeros((0,))
        violation_parts = []
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            image_background = final_scores[batch_index, selected[batch_index]].amax(-1)
            if not source_indices.numel() or not image_background.numel():
                continue
            labels = targets[batch_index]['labels'][target_indices]
            image_positive = final_scores[batch_index, source_indices, labels]
            k = min(image_positive.numel(), image_background.numel())
            violation_parts.append(
                (image_background.topk(k).values + self.final_bg_margin
                 > image_positive.topk(k, largest=False).values).float()
            )
        violation = torch.cat(violation_parts).mean() if violation_parts else zero
        self._diagnostics.update({
            'metric_final_bg_selected_per_image': counts.mean() if counts.numel() else zero,
            'metric_final_bg_score_mean': background_values.mean() if background_values.numel() else zero,
            'metric_matched_final_score_mean': positive_values.mean() if positive_values.numel() else zero,
            'metric_rank_violation_ratio': violation,
        })
        return {'loss_final_bg_rank': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self._alignment_cache = {}
        self._diagnostics = {}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'loc_quality': self.loss_loc_quality,
            'defect': self.loss_defect,
            'query_semantic': self.loss_query_semantic,
            'final_bg_rank': self.loss_final_bg_rank,
            'local': self.loss_local,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            indices_go = indices
            num_boxes_go = None

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        if num_boxes_go is None:
            num_boxes_go = num_boxes

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self._branch_losses('main', outputs):
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self._branch_losses('aux', aux_outputs):
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self._branch_losses('legacy', aux_outputs):
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self._branch_losses('legacy', aux_outputs):
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self._branch_losses('legacy', aux_outputs):
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self._branch_losses('legacy', aux_outputs):
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        losses.update({key: value.detach() for key, value in self._diagnostics.items()})
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
