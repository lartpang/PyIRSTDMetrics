import numpy as np
from scipy import optimize
from skimage import measure

from .utils import divide_func


class ProbabilityDetectionAndFalseAlarmRate:
    """Target-wise Probability of Detection (PD) and False Alarm Rate (Fa)"""

    def __init__(self, num_bins=1, distance_threshold=3):
        super().__init__()
        assert distance_threshold > 0, f"distance_threshold={distance_threshold} must be greater than 0."
        assert isinstance(num_bins, int) and num_bins > 0, f"int num_bins={num_bins} must be greater than 0."

        self.distance_threshold = distance_threshold
        if num_bins == 1:
            self.thresholds = np.array([0.5])
        else:
            self.thresholds = np.linspace(0, 1, num_bins, endpoint=False)

        self.tp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fn_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_area = np.zeros(shape=num_bins, dtype=int)
        self.total_area = 0

    def update(self, prob: np.ndarray, mask: np.ndarray):
        """Update the prob and mask to calculate the target-wise positive detection.

        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary bin_mask.
        """
        assert prob.shape == mask.shape, (prob.shape, mask.shape)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, mask.dtype
        self.total_area += mask.size

        mask_tgts = measure.regionprops(measure.label(mask.astype(int), connectivity=2))
        num_mask_tgts = len(mask_tgts)

        # >0, >1bin, >2bin, ..., >(num_bins-1)*bin
        for idx_bin, thr in enumerate(self.thresholds):
            bin_prob = prob > thr
            prob_tgts = measure.regionprops(measure.label(bin_prob.astype(int), connectivity=2))
            num_prob_tgts = len(prob_tgts)

            # idx_prob_tgt -> idx_mask_tgt
            matched_status = np.zeros((num_mask_tgts, num_prob_tgts), dtype=bool)
            for idx_mask_tgt, mask_tgt in enumerate(mask_tgts):
                cnt_mask_xy = np.asarray(mask_tgt.centroid)
                for idx_prob_tgt, prob_tgt in enumerate(prob_tgts):
                    if np.any(matched_status[:, idx_prob_tgt]):
                        continue

                    cnt_prob_xy = np.asarray(prob_tgt.centroid)
                    distance = np.linalg.norm(cnt_prob_xy - cnt_mask_xy)
                    if distance < self.distance_threshold:
                        matched_status[idx_mask_tgt, idx_prob_tgt] = True
                        break

            tp_objs = np.count_nonzero(matched_status)
            fp_objs = num_prob_tgts - tp_objs
            fn_objs = num_mask_tgts - tp_objs

            matched_prob_tgt_status = np.count_nonzero(matched_status, axis=0)
            unmatched_pre_tgt_indices = np.where(matched_prob_tgt_status == 0)[0]
            fp_area = sum([prob_tgts[j].area if num_prob_tgts > 0 else 0 for j in unmatched_pre_tgt_indices])

            self.tp_objs[idx_bin] += tp_objs
            self.fp_objs[idx_bin] += fp_objs
            self.fn_objs[idx_bin] += fn_objs
            self.fp_area[idx_bin] += fp_area

    def get(self) -> dict:
        """Return the target-wise metrics: probability of detection and false alarm rate.

        Returns:
            {
                probability_detection (np.ndarray): probability_detection, (N,)
                false_alarm (np.ndarray): false_alarm, (N,)
            }
        """
        probability_detection = divide_func(self.tp_objs, self.tp_objs + self.fn_objs)
        false_alarm = divide_func(self.fp_area, [self.total_area])
        return {"probability_detection": probability_detection, "false_alarm": false_alarm}


class ShootingRuleBasedProbabilityDetectionAndFalseAlarmRate:
    """Target-wise Probability of Detection (PD) and False Alarm Rate (Fa) based on the shooting rule [1] (https://figures.semanticscholar.org/12171bbd9edde3bbaa99880f3a947e22efe43188/7-Figure5-1.png).

    References:
        [1] Li, Ruojing et al. "Direction-Coded Temporal U-Shape Module for Multiframe Infrared Small Target Detection." IEEE Transactions on Neural Networks and Learning Systems 36 (2023): 555-568.
    """

    def __init__(self, num_bins=1, box_1_radius=1, box_2_radius=4):
        super().__init__()
        assert box_2_radius > box_1_radius > 0, (
            f"box_2_radius={box_2_radius} must be greater than box_1_radius={box_1_radius} and greater than 0."
        )
        assert isinstance(num_bins, int) and num_bins > 0, f"int num_bins={num_bins} must be greater than 0."

        self.box_1_radius = box_1_radius
        self.box_2_radius = box_2_radius
        if num_bins == 1:
            self.thresholds = np.array([0.5])
        else:
            self.thresholds = np.linspace(0, 1, num_bins, endpoint=False)

        self.tp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_area = np.zeros(shape=num_bins, dtype=int)
        self.total_objs = 0
        self.total_area = 0

    def update(self, prob: np.ndarray, mask: np.ndarray):
        """Update the prob and mask to calculate the target-wise positive detection.

        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary bin_mask.
        """
        assert prob.shape == mask.shape, (prob.shape, mask.shape)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, mask.dtype
        self.total_area += mask.size

        mask_tgts = measure.regionprops(measure.label(mask, connectivity=2))
        self.total_objs += len(mask_tgts)

        invalid_mask_region = np.ones(prob.shape, dtype=bool)
        # >0, >1bin, >2bin, ..., >(num_bins-1)*bin
        for idx_bin, thr in enumerate(self.thresholds):
            # the original code uses `>=`, but here, we use `>` to keep the same setting as other codes
            bin_prob = prob > thr

            invalid_mask_region.fill(True)
            for mask_tgt in mask_tgts:
                is_matched = False

                for h, w in mask_tgt.coords:
                    h1, h2 = max(0, h - self.box_2_radius), min(prob.shape[0], h + self.box_2_radius + 1)
                    w1, w2 = max(0, w - self.box_2_radius), min(prob.shape[1], w + self.box_2_radius + 1)
                    invalid_mask_region[h1:h2, w1:w2] = False

                    if not is_matched:
                        h1, h2 = max(0, h - self.box_1_radius), min(prob.shape[0], h + self.box_1_radius + 1)
                        w1, w2 = max(0, w - self.box_1_radius), min(prob.shape[1], w + self.box_1_radius + 1)
                        if np.count_nonzero(bin_prob[h1:h2, w1:w2]) >= 1:
                            is_matched = True

                if is_matched:
                    self.tp_objs[idx_bin] += 1
            self.fp_area[idx_bin] += np.count_nonzero(bin_prob & invalid_mask_region)

    def get(self) -> dict:
        """Return the target-wise metrics: probability of detection and false alarm rate.

        Returns:
            {
                probability_detection (np.ndarray): probability_detection, (N,)
                false_alarm (np.ndarray): false_alarm, (N,)
            }
        """
        probability_detection = divide_func(self.tp_objs, self.total_objs)
        false_alarm = divide_func(self.fp_area, [self.total_area])
        return {"probability_detection": probability_detection, "false_alarm": false_alarm}


class OPDCMatching:
    def __init__(self, overlap_threshold=0.5, distance_threshold=3):
        if not (isinstance(overlap_threshold, float) and 0 <= overlap_threshold < 1):
            raise ValueError(f"float overlap_threshold={overlap_threshold} must be in [0, 1].")
        if not (isinstance(distance_threshold, int) and distance_threshold > 0):
            raise ValueError(f"int distance_threshold={distance_threshold} must be greater than 0.")
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold

    def get_paired_info(self, pred_tgts_map, mask_tgts_map, pred_props, mask_props):
        num_pred_tgts = len(pred_props)
        num_mask_tgts = len(mask_props)

        paired_distance = np.empty(shape=(max(num_mask_tgts, 1), max(num_pred_tgts, 1)), dtype=float)
        paired_distance.fill(mask_tgts_map.size)
        paired_iou = np.zeros(shape=(max(num_mask_tgts, 1), max(num_pred_tgts, 1)), dtype=float)
        for i, mask_tgt in enumerate(mask_props):
            mask_cnt_yx = np.asarray(mask_tgt.centroid).reshape(2)

            for j, pred_tgt in enumerate(pred_props):
                pred_cnt_yx = np.asarray(pred_tgt.centroid).reshape(2)
                paired_distance[i, j] = np.linalg.norm(pred_cnt_yx - mask_cnt_yx)

                _pred_tgt_mask = pred_tgts_map == pred_tgt.label
                _mask_tgt_mask = mask_tgts_map == mask_tgt.label
                _inter_area = np.count_nonzero(_pred_tgt_mask & _mask_tgt_mask)

                iou = 0
                if _inter_area > 0:
                    _union_area = np.count_nonzero(_pred_tgt_mask | _mask_tgt_mask)
                    iou = _inter_area / _union_area
                paired_iou[i, j] = iou
        return paired_distance, paired_iou

    def overlap_priority_constraint(self, matched_status, paired_distance, valid_iou_status):
        row_ind, col_ind = optimize.linear_sum_assignment(paired_distance)
        matched_status[row_ind, col_ind] = True
        matched_status &= valid_iou_status

    def distance_based_compensation(self, matched_status, paired_distance, valid_distance_status):
        _unmatched_mask_tgt_idxs = np.where(matched_status.sum(axis=1) == 0)[0]
        _unmatched_pred_tgt_idxs = np.where(matched_status.sum(axis=0) == 0)[0]
        sub_paired_distance = paired_distance[_unmatched_mask_tgt_idxs, :][:, _unmatched_pred_tgt_idxs]
        row_ind, col_ind = optimize.linear_sum_assignment(sub_paired_distance)
        for row_idx, col_idx in zip(row_ind, col_ind):
            ori_row_idx = _unmatched_mask_tgt_idxs[row_idx]
            ori_col_idx = _unmatched_pred_tgt_idxs[col_idx]
            if valid_distance_status[ori_row_idx, ori_col_idx]:
                matched_status[ori_row_idx, ori_col_idx] = True

    def __call__(self, pred_tgts_map, mask_tgts_map, pred_props, mask_props):
        paired_distance, paired_iou = self.get_paired_info(pred_tgts_map, mask_tgts_map, pred_props, mask_props)

        # mask_tgt (col axis) -> pred_tgt (row axis)
        valid_iou_status = paired_iou >= self.overlap_threshold
        valid_distance_status = paired_distance < self.distance_threshold

        matched_status = np.zeros_like(paired_distance, dtype=bool)
        if np.count_nonzero(valid_iou_status) > 0:
            self.overlap_priority_constraint(matched_status, paired_distance, valid_iou_status)
        if np.count_nonzero(valid_distance_status) > 0:
            self.distance_based_compensation(matched_status, paired_distance, valid_distance_status)
        return matched_status


class DistanceOnlyMatching:
    def __init__(self, distance_threshold=3):
        if not (isinstance(distance_threshold, int) and distance_threshold > 0):
            raise ValueError(f"int distance_threshold={distance_threshold} must be greater than 0.")
        self.distance_threshold = distance_threshold

    def __call__(self, pred_tgts_map, mask_tgts_map, pred_props, mask_props):
        num_pred_tgts = len(pred_props)
        num_mask_tgts = len(mask_props)

        # idx_pred_tgt -> idx_mask_tgt
        matched_status = np.zeros((num_mask_tgts, num_pred_tgts), dtype=bool)
        for idx_mask_tgt, mask_tgt in enumerate(mask_props):
            cnt_mask_xy = np.asarray(mask_tgt.centroid)
            for idx_pred_tgt, pred_tgt in enumerate(pred_props):
                if np.any(matched_status[:, idx_pred_tgt]):
                    continue

                cnt_pred_xy = np.asarray(pred_tgt.centroid)
                distance = np.linalg.norm(cnt_pred_xy - cnt_mask_xy)
                if distance < self.distance_threshold:
                    matched_status[idx_mask_tgt, idx_pred_tgt] = True
                    break
        return matched_status


class MatchingBasedMetrics:
    def __init__(self, num_bins=1, *, matching_method=OPDCMatching(overlap_threshold=0.5, distance_threshold=3)):
        super().__init__()
        if not (isinstance(num_bins, int) and num_bins > 0):
            raise ValueError(f"int num_bins={num_bins} must be greater than 0.")

        if num_bins == 1:
            self.thresholds = np.array([0.5])
        else:
            self.thresholds = np.linspace(0, 1, num_bins, endpoint=False)

        self.tp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fn_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_area = np.zeros(shape=num_bins, dtype=int)
        self.tp_ious = np.zeros(shape=num_bins, dtype=float)
        self.total_area = 0
        self.matching_method = matching_method

    def update(self, prob: np.ndarray, mask: np.ndarray):
        """Update the prob and mask to calculate the target-wise positive detection.

        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary mask.
        """
        assert prob.shape == mask.shape, (prob.shape, mask.shape)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, mask.dtype
        self.total_area += mask.size

        for idx_bin, thr in enumerate(self.thresholds):
            pred_tgts_map = measure.label(prob > thr, connectivity=2)
            mask_tgts_map = measure.label(mask, connectivity=2)
            pred_props = measure.regionprops(pred_tgts_map)
            mask_props = measure.regionprops(mask_tgts_map)
            num_pred_tgts = len(pred_props)
            num_mask_tgts = len(mask_props)

            matched_status = self.matching_method(pred_tgts_map, mask_tgts_map, pred_props, mask_props)
            matched_pred_tgt_status = np.count_nonzero(matched_status, axis=0)
            if not set(matched_pred_tgt_status.tolist()).issubset((0, 1)):
                raise ValueError("Some predicted targets are matched to multiple GT targets.")
            matched_mask_tgt_status = np.count_nonzero(matched_status, axis=1)
            if not set(matched_mask_tgt_status.tolist()).issubset((0, 1)):
                raise ValueError("Some GT targets are matched to multiple predicted targets.")

            tp_ious = []
            h_indices, w_indices = np.where(matched_status)
            for h_index, w_index in zip(h_indices, w_indices):
                if num_mask_tgts > 0:
                    mask_tgt_mask = mask_tgts_map == mask_props[h_index].label
                else:
                    mask_tgt_mask = mask_tgts_map  # all zero, background
                if num_pred_tgts > 0:
                    pred_tgt_mask = pred_tgts_map == pred_props[w_index].label
                else:
                    pred_tgt_mask = pred_tgts_map  # background

                tp = np.count_nonzero(pred_tgt_mask & mask_tgt_mask)  # tp
                fp = np.count_nonzero(pred_tgt_mask & ~mask_tgt_mask)
                fn = np.count_nonzero(~pred_tgt_mask & mask_tgt_mask)
                tp_fp_fn = np.count_nonzero(pred_tgt_mask | mask_tgt_mask)  # tp + fp + fn
                assert tp_fp_fn == tp + fp + fn, tp_fp_fn - (tp + fp + fn)
                tp_ious.append(divide_func(tp, tp_fp_fn))
            self.tp_ious[idx_bin] += sum(tp_ious)

            num_final_unassigned_pred_tgts = np.count_nonzero(matched_pred_tgt_status == 0)
            num_final_unassigned_mask_tgts = np.count_nonzero(matched_mask_tgt_status == 0)

            self.tp_objs[idx_bin] += np.count_nonzero(matched_status)
            self.fp_objs[idx_bin] += num_final_unassigned_pred_tgts
            self.fn_objs[idx_bin] += num_final_unassigned_mask_tgts
            unmatched_pred_tgt_indices = np.where(matched_pred_tgt_status == 0)[0]

            if num_pred_tgts > 0:
                self.fp_area[idx_bin] += sum([pred_props[j].area for j in unmatched_pred_tgt_indices])

    def get(self):
        """Return all average metrics.

        Returns:
            np.ndarray: false_alarm
            np.ndarray: probability_detection
            np.ndarray: seg_iou
            np.ndarray: loc_iou
        """
        false_alarm = divide_func(self.fp_area, [self.total_area])
        probability_detection = divide_func(self.tp_objs, self.tp_objs + self.fn_objs)
        seg_iou = divide_func(self.tp_ious, self.tp_objs)
        loc_iou = divide_func(self.tp_objs, self.tp_objs + self.fp_objs + self.fn_objs)
        hiou = seg_iou * loc_iou
        return dict(
            false_alarm=false_alarm,
            probability_detection=probability_detection,
            hiou=hiou,
        )


class HierarchicalIoUBasedErrorAnalysis:
    def __init__(self, num_bins=1, overlap_threshold=0.5, distance_threshold=3):
        if not (isinstance(num_bins, int) and num_bins > 0):
            raise ValueError(f"int num_bins={num_bins} must be greater than 0.")

        if num_bins == 1:
            self.thresholds = np.array([0.5])
        else:
            self.thresholds = np.linspace(0, 1, num_bins, endpoint=False)

        self.tp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_objs = np.zeros(shape=num_bins, dtype=int)
        self.fn_objs = np.zeros(shape=num_bins, dtype=int)
        self.fp_area = np.zeros(shape=num_bins, dtype=int)
        self.tp_ious = np.zeros(shape=num_bins, dtype=float)
        self.total_area = 0

        # pixel-wise err
        self.seg_mrg_err = np.zeros(shape=num_bins, dtype=float)
        self.seg_itf_err = np.zeros(shape=num_bins, dtype=float)
        self.seg_pcp_err = np.zeros(shape=num_bins, dtype=float)
        # target-wise err
        self.num_tgts_for_itf_err = np.zeros(shape=num_bins, dtype=int)
        self.num_tgts_for_pcp_err = np.zeros(shape=num_bins, dtype=int)
        self.num_tgts_for_m2s_err = np.zeros(shape=num_bins, dtype=int)
        self.num_tgts_for_s2m_err = np.zeros(shape=num_bins, dtype=int)
        self.matching_method = OPDCMatching(overlap_threshold=overlap_threshold, distance_threshold=distance_threshold)

    def update(self, prob: np.ndarray, mask: np.ndarray):
        """Update the prob and mask to calculate the matching-based metrics.

        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary mask.
        """
        assert prob.shape == mask.shape, (prob.shape, mask.shape)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, mask.dtype

        for idx_bin, thr in enumerate(self.thresholds):
            pred_tgts_map = measure.label(prob > thr, connectivity=2)
            pred_props = measure.regionprops(pred_tgts_map)
            mask_tgts_map = measure.label(mask, connectivity=2)
            mask_props = measure.regionprops(mask_tgts_map)
            num_pred_tgts = len(pred_props)
            num_mask_tgts = len(mask_props)

            paired_distance, paired_iou = self.matching_method.get_paired_info(
                pred_tgts_map, mask_tgts_map, pred_props, mask_props
            )
            valid_iou_status = paired_iou >= self.matching_method.overlap_threshold
            valid_distance_status = paired_distance < self.matching_method.distance_threshold
            init_matched_status = valid_iou_status | valid_distance_status

            matched_status = np.zeros_like(paired_distance, dtype=bool)
            self.matching_method.overlap_priority_constraint(matched_status, paired_distance, valid_iou_status)
            self.matching_method.distance_based_compensation(matched_status, paired_distance, valid_distance_status)

            matched_pred_tgt_status = np.count_nonzero(matched_status, axis=0)
            if not set(matched_pred_tgt_status.tolist()).issubset((0, 1)):
                raise ValueError("Some predicted targets are matched to multiple GT targets.")
            matched_mask_tgt_status = np.count_nonzero(matched_status, axis=1)
            if not set(matched_mask_tgt_status.tolist()).issubset((0, 1)):
                raise ValueError("Some GT targets are matched to multiple predicted targets.")

            matched_ious = []
            seg_mrg_err = []
            seg_itf_err = []
            seg_pcp_err = []
            h_indices, w_indices = np.where(matched_status)
            for h_index, w_index in zip(h_indices, w_indices):
                if num_mask_tgts > 0:
                    mask_tgt_mask = mask_tgts_map == mask_props[h_index].label
                    other_mask_tgts = mask ^ mask_tgt_mask
                else:
                    mask_tgt_mask = mask_tgts_map  # all zero, background
                if num_pred_tgts > 0:
                    pred_tgt_mask = pred_tgts_map == pred_props[w_index].label
                else:
                    pred_tgt_mask = pred_tgts_map  # background

                tp = np.count_nonzero(pred_tgt_mask & mask_tgt_mask)

                fp_tgts = pred_tgt_mask & ~mask_tgt_mask
                fp_from_other_tgts = fp_tgts & other_mask_tgts
                fp_from_self_tgt = fp_tgts ^ fp_from_other_tgts
                fp = np.count_nonzero(fp_tgts)
                fp_from_other_tgts = np.count_nonzero(fp_from_other_tgts)
                fp_from_self_tgt = np.count_nonzero(fp_from_self_tgt)

                fn = np.count_nonzero(~pred_tgt_mask & mask_tgt_mask)
                tp_fp_fn = np.count_nonzero(pred_tgt_mask | mask_tgt_mask)  # tp + fp + fn

                assert tp_fp_fn == tp + fp + fn, tp_fp_fn - (tp + fp + fn)
                matched_ious.append(divide_func(tp, tp_fp_fn))
                seg_mrg_err.append(divide_func(fp_from_other_tgts, tp_fp_fn))
                seg_itf_err.append(divide_func(fp_from_self_tgt, tp_fp_fn))
                seg_pcp_err.append(divide_func(fn, tp_fp_fn))
            self.tp_ious[idx_bin] += sum(matched_ious)
            self.seg_mrg_err[idx_bin] += sum(seg_mrg_err)
            self.seg_itf_err[idx_bin] += sum(seg_itf_err)
            self.seg_pcp_err[idx_bin] += sum(seg_pcp_err)

            # Interference error and Perception error: lbl -> pre
            num_init_unassigned_mask_tgts = np.count_nonzero(init_matched_status.sum(axis=1) == 0)
            num_init_unassigned_pred_tgts = np.count_nonzero(init_matched_status.sum(axis=0) == 0)

            self.num_tgts_for_itf_err[idx_bin] += num_init_unassigned_pred_tgts
            self.num_tgts_for_pcp_err[idx_bin] += num_init_unassigned_mask_tgts

            num_final_unassigned_pred_tgts = np.count_nonzero(matched_pred_tgt_status == 0)
            num_final_unassigned_mask_tgts = np.count_nonzero(matched_mask_tgt_status == 0)

            # Assignment error (m2s, s2m)
            self.num_tgts_for_m2s_err[idx_bin] += num_final_unassigned_pred_tgts - num_init_unassigned_pred_tgts
            self.num_tgts_for_s2m_err[idx_bin] += num_final_unassigned_mask_tgts - num_init_unassigned_mask_tgts

            self.tp_objs[idx_bin] += np.count_nonzero(matched_status)
            self.fp_objs[idx_bin] += num_final_unassigned_pred_tgts
            self.fn_objs[idx_bin] += num_final_unassigned_mask_tgts

    def get(self):
        """Return all average metrics.

        Returns:
            np.ndarray: seg_iou
            np.ndarray: seg_mrg_err
            np.ndarray: seg_itf_err
            np.ndarray: seg_pcp_err
            np.ndarray: loc_iou
            np.ndarray: loc_s2m_err
            np.ndarray: loc_m2s_err
            np.ndarray: loc_itf_err
            np.ndarray: loc_pcp_err
        """
        # pixel-wise segmentation quality
        seg_iou = divide_func(self.tp_ious, self.tp_objs)
        seg_itf_err = divide_func(self.seg_itf_err, self.tp_objs)
        seg_mrg_err = divide_func(self.seg_mrg_err, self.tp_objs)
        seg_pcp_err = divide_func(self.seg_pcp_err, self.tp_objs)

        # target-wise recognition quality
        denominator = self.tp_objs + self.fp_objs + self.fn_objs
        loc_iou = divide_func(self.tp_objs, denominator)
        loc_itf_err = divide_func(self.num_tgts_for_itf_err, denominator)
        loc_m2s_err = divide_func(self.num_tgts_for_m2s_err, denominator)
        loc_pcp_err = divide_func(self.num_tgts_for_pcp_err, denominator)
        loc_s2m_err = divide_func(self.num_tgts_for_s2m_err, denominator)

        return dict(
            seg_iou=seg_iou,
            seg_mrg_err=seg_mrg_err,
            seg_itf_err=seg_itf_err,
            seg_pcp_err=seg_pcp_err,
            loc_iou=loc_iou,
            loc_s2m_err=loc_s2m_err,
            loc_m2s_err=loc_m2s_err,
            loc_itf_err=loc_itf_err,
            loc_pcp_err=loc_pcp_err,
        )
