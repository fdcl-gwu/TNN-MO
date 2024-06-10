"""
Default DETR model (v1.1) and criterion classes.
"""
print("DETR 1.1 network called")
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
					   accuracy, get_world_size, interpolate,
					   is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
						   dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from util.cr import cr_loss


class DETR(nn.Module):
	""" This is the DETR module that performs object detection """
	def __init__(self, backbone, transformer, num_classes, num_queries, num_keys, aux_loss=False):
		""" Initializes the model.
		Parameters:
			backbone: torch module of the backbone to be used. See backbone.py
			transformer: torch module of the transformer architecture. See transformer.py
			num_classes: number of object classes
			num_queries: number of object queries, ie detection slot. This is the maximal number of objects
						 DETR can detect in a single image. For COCO, we recommend 100 queries.
			aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
		"""
		super().__init__()
		self.num_queries = num_queries
		self.transformer = transformer
		hidden_dim = transformer.d_model
		self.num_keys = num_keys
		num_body_parts = (self.num_keys*2) # 32  (del_x1, del_y1, vis) * 32 + 2  # 17  (del_x1, del_y1, vis) * 17 + 2 
		self.class_embed_out = nn.Linear(hidden_dim, num_classes + 1)
		self.pose_embed = MLP(hidden_dim, hidden_dim, num_body_parts, 3)
		self.query_embed = nn.Embedding(num_queries, hidden_dim)
		self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
		self.backbone = backbone
		self.aux_loss = aux_loss


	def forward(self, samples: NestedTensor):
		""" The forward expects a NestedTensor, which consists of:
			   - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
			   - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

			It returns a dict with the following elements:
			   - "pred_logits": the classification logits (including no-object) for all queries.
								Shape= [batch_size x num_queries x (num_classes + 1)]
			   - "pred_boxes": The normalized boxes coordinates for all queries, represented as
							   (center_x, center_y, height, width). These values are normalized in [0, 1],
							   relative to the size of each individual image (disregarding possible padding).
							   See PostProcess for information on how to retrieve the unnormalized bounding box.
			   - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
								dictionnaries containing the two above keys for each decoder layer.
		"""
		if isinstance(samples, (list, torch.Tensor)):
			samples = nested_tensor_from_tensor_list(samples)
		features, pos = self.backbone(samples)

		src, mask = features[-1].decompose()
		assert mask is not None
	
		hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

		outputs_class = self.class_embed_out(hs)
		outputs_keypoints = self.pose_embed(hs)	

		out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints[-1]}
		if self.aux_loss:
			out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_keypoints)
		return out

	@torch.jit.unused
	def _set_aux_loss(self, outputs_class, outputs_keypoints):
		# this is a workaround to make torchscript happy, as torchscript
		# doesn't support dictionary with non-homogeneous values, such
		# as a dict having both a Tensor and a list.
		return [{'pred_logits': a, 'pred_keypoints': b}
				for a, b in zip(outputs_class[:-1], outputs_keypoints[:-1])]


class SetCriterion(nn.Module):
	""" This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""
	def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_keys):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		empty_weight = torch.ones(self.num_classes + 1)
		empty_weight[-1] = self.eos_coef
		self.register_buffer('empty_weight', empty_weight)
		self.num_keys = num_keys
		self.l_deltas = 0.5
		self.l_vis = 0.2
		self.l_ctr = 0.5
		self.gamma = 10
		self.delta = 0.0
		self.target_cr = 4/3

	def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert 'pred_logits' in outputs
		src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = torch.full(src_logits.shape[:2], self.num_classes,
									dtype=torch.int64, device=src_logits.device)

		
		target_classes[idx] = target_classes_o #*(-1)

		loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
	
		losses = {'loss_ce': loss_ce}
		
		if log:
			# TODO this should probably be a separate loss, not hacked in this one here
			losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
		return losses

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets, indices, num_boxes):
		""" Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
		This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
		"""
		pred_logits = outputs['pred_logits']
		device = pred_logits.device
		tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
		# Count the number of predictions that are NOT "no-object" (which is the last class)
		card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
		card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
		losses = {'cardinality_error': card_err}
		return losses

	def loss_keypoints(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the bounding boxes, the L1 regression loss
		   targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		   The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		"""
		target_cr = self.target_cr
		#assert 'pred_boxes' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_keypoints = outputs['pred_keypoints'][idx]

		Z_pred = src_keypoints[:, :self.num_keys*2]
	
		targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)

		Z_gt = targets_keypoints[:, :self.num_keys*2]

		abs_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')

		total_keypoints_loss = self.gamma * torch.sum(abs_loss)
		
		# CR_loss = cr_loss(Z_pred,target_cr)
		# total_CR_loss = self.delta * CR_loss

		losses = {}
		losses['loss_keypoints'] = total_keypoints_loss / num_boxes #+ total_CR_loss / num_boxes
		#losses['loss_cr'] = total_CR_loss / num_boxes
		return losses

	def loss_masks(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the masks: the focal loss and the dice loss.
		   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
		"""
		assert "pred_masks" in outputs

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = self._get_tgt_permutation_idx(indices)
		src_masks = outputs["pred_masks"]
		src_masks = src_masks[src_idx]
		masks = [t["masks"] for t in targets]
		# TODO use valid to mask invalid areas due to padding in loss
		target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
		target_masks = target_masks.to(src_masks)
		target_masks = target_masks[tgt_idx]

		# upsample predictions to the target size
		src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
								mode="bilinear", align_corners=False)
		src_masks = src_masks[:, 0].flatten(1)

		target_masks = target_masks.flatten(1)
		target_masks = target_masks.view(src_masks.shape)
		losses = {
			"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
			"loss_dice": dice_loss(src_masks, target_masks, num_boxes),
		}
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

	def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			'cardinality': self.loss_cardinality,
			'keypoints': self.loss_keypoints,
			#'masks': self.loss_masks
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

	def forward(self, outputs, targets):
		""" This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

		
		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_boxes = sum(len(t["labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_boxes)
		num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets, num_boxes)

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

		# In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
		if 'aux_outputs' in outputs:
			for i, aux_outputs in enumerate(outputs['aux_outputs']):
				indices = self.matcher(aux_outputs, targets, num_boxes)
				for loss in self.losses:
					if loss == 'masks':
						# Intermediate masks losses are too costly to compute, we ignore them.
						continue
					kwargs = {}
					if loss == 'labels':
						# Logging is enabled only for the last layer
						kwargs = {'log': False}
					l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
					l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
					losses.update(l_dict)

		return losses


class PostProcess(nn.Module):
	""" This module converts the model's output into the format expected by the coco api"""
	@torch.no_grad()
	def forward(self, outputs, target_sizes):
		""" Perform the computation
		Parameters:
			outputs: raw outputs of the model
			target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
						  For evaluation, this must be the original image size (before any data augmentation)
						  For visualization, this should be the image size after data augment, but before padding
		"""
		out_logits, out_keypoints = outputs['pred_logits'], outputs['pred_keypoints']

		assert len(out_logits) == len(target_sizes)
		assert target_sizes.shape[1] == 2

		prob = F.softmax(out_logits, -1)
		scores, labels = prob[..., :-1].max(-1)

		# convert to [x0, y0, x1, y1] format
		keypoints = box_ops.box_cxcywh_to_xyxy(out_keypoints)
		# and from relative [0, 1] to absolute [0, height] coordinates
		img_h, img_w = target_sizes.unbind(1)
		scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
		keypoints = keypoints * scale_fct[:, None, :]

		results = [{'scores': s, 'labels': l, 'keypoints': b} for s, l, b in zip(scores, labels, keypoints)]

		return results


class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


def build(args):
	num_classes = args.num_classes

	device = torch.device(args.device)

	backbone = build_backbone(args)

	transformer = build_transformer(args)

	model = DETR(
		backbone,
		transformer,
		num_classes=num_classes,
		num_queries=args.num_queries,
		aux_loss=args.aux_loss,
		num_keys = args.num_keys
	)
	if args.masks:
		model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
	matcher = build_matcher(args)
	weight_dict = {'loss_ce': 1, 'loss_keypoints': 1}
	weight_dict['loss_giou'] = args.giou_loss_coef
	if args.masks:
		weight_dict["loss_mask"] = args.mask_loss_coef
		weight_dict["loss_dice"] = args.dice_loss_coef

	if args.aux_loss:
		aux_weight_dict = {}
		for i in range(args.dec_layers - 1):
			aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
		weight_dict.update(aux_weight_dict)

	losses = ['labels', 'keypoints', 'cardinality']
	if args.masks:
		losses += ["masks"]
	criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
							 eos_coef=args.eos_coef, losses=losses, num_keys = args.num_keys)
	criterion.to(device)
	postprocessors = {'keypoints': PostProcess()}
	if args.masks:
		postprocessors['segm'] = PostProcessSegm()
		if args.dataset_file == "coco_panoptic":
			is_thing_map = {i: i <= 90 for i in range(201)}
			postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

	return model, criterion, postprocessors