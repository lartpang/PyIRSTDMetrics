# -*- coding: utf-8 -*-
# @Time    : 2021/1/4
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys

import cv2
import numpy as np

sys.path.append("..")

from py_irstd_metrics.pixelwise_metrics import (
    CMMetrics,
    FmeasureHandler,
    FPRHandler,
    IoUHandler,
    PrecisionHandler,
    RecallHandler,
    TPRHandler,
)
from py_irstd_metrics.targetwise_metrics import (
    DistanceOnlyMatching,
    HierarchicalIoUBasedErrorAnalysis,
    MatchingBasedMetrics,
    OPDCMatching,
    ProbabilityDetectionAndFalseAlarmRate,
)
from py_irstd_metrics.utils import ndarray_to_basetype


class BasicIRSTDPerformance:
    def __init__(self, num_bins=10) -> None:
        self.original_pd_fa = ProbabilityDetectionAndFalseAlarmRate(
            num_bins=num_bins,
            distance_threshold=3,
        )

    def update(self, prob: np.ndarray, mask: np.ndarray, mask_path: str = None) -> None:
        """
        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary bin_mask.
            mask_path (str, optional): the patch of the mask file. Defaults to None.
        """
        assert prob.shape == mask.shape, (prob.shape, mask.shape, mask_path)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, (mask.dtype, mask_path)

        self.original_pd_fa.update(prob=prob, mask=mask)

    def get_all_results(self, num_bits=4):
        original_pd_fa = self.original_pd_fa.get()

        return {
            # target-level
            "pd": original_pd_fa["probability_detection"].round(num_bits),
            "fa": (original_pd_fa["false_alarm"] * 1e6).round(num_bits),
        }

    def show(self, num_bits=4, return_ndarray: bool = False) -> dict:
        results = self.get_all_results(num_bits)
        if not return_ndarray:
            results = ndarray_to_basetype(results)
        return results


class IRSTDPerformanceAnalysis:
    def __init__(self, num_bins=10) -> None:
        self.pixel_level_metrics = CMMetrics(
            threshold=0.5,  # for binary metric
            num_bins=num_bins,
            metric_handlers={
                # values
                "iou": IoUHandler(with_dynamic=False, with_binary=True, sample_based=False),
                "normalized_iou": IoUHandler(with_dynamic=False, with_binary=True, sample_based=True),
                "f1": FmeasureHandler(with_dynamic=False, with_binary=True, sample_based=False, beta=1),
                # curves
                "precision": PrecisionHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "recall": RecallHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "TPR": TPRHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "FPR": FPRHandler(with_dynamic=True, with_binary=False, sample_based=False),
            },
        )
        self.opdc_based_metrics = MatchingBasedMetrics(
            num_bins=num_bins,
            matching_method=OPDCMatching(overlap_threshold=0.5, distance_threshold=3),
        )
        self.distance_based_metrics = MatchingBasedMetrics(
            num_bins=num_bins,
            matching_method=DistanceOnlyMatching(distance_threshold=3),
        )
        self.hiou_based_errors = HierarchicalIoUBasedErrorAnalysis(
            num_bins=num_bins,
            overlap_threshold=0.5,
            distance_threshold=3,
        )

    def update(self, prob: np.ndarray, mask: np.ndarray, mask_path: str = None) -> None:
        """
        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary bin_mask.
            mask_path (str, optional): the patch of the mask file. Defaults to None.
        """
        assert prob.shape == mask.shape, (prob.shape, mask.shape, mask_path)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, (mask.dtype, mask_path)

        self.pixel_level_metrics.update(prob=prob, mask=mask)
        self.opdc_based_metrics.update(prob=prob, mask=mask)
        self.distance_based_metrics.update(prob=prob, mask=mask)
        self.hiou_based_errors.update(prob=prob, mask=mask)

    def get_all_results(self, num_bits=4):
        pixel_level_metrics = self.pixel_level_metrics.get()
        opdc_based_metrics = self.opdc_based_metrics.get()
        distance_based_metrics = self.distance_based_metrics.get()
        hiou_based_errors = self.hiou_based_errors.get()

        return {
            # pixel-level
            "iou": pixel_level_metrics["iou"]["binary"].round(num_bits),
            "niou": pixel_level_metrics["normalized_iou"]["binary"].round(num_bits),
            "f1": pixel_level_metrics["f1"]["binary"].round(num_bits),
            # target-level
            "pd_distonly": distance_based_metrics["probability_detection"].round(num_bits),
            "pd_opdc": opdc_based_metrics["probability_detection"].round(num_bits),
            "fa_distonly": (distance_based_metrics["false_alarm"] * 1e6).round(num_bits),
            "fa_opdc": (opdc_based_metrics["false_alarm"] * 1e6).round(num_bits),
            # hybrid-level
            "hiou_opdc": (opdc_based_metrics["hiou"]).round(num_bits),
            # error_analysis
            "seg_iou": hiou_based_errors["seg_iou"].round(num_bits),
            "seg_mrg_err": hiou_based_errors["seg_mrg_err"].round(num_bits),
            "seg_itf_err": hiou_based_errors["seg_itf_err"].round(num_bits),
            "seg_pcp_err": hiou_based_errors["seg_pcp_err"].round(num_bits),
            "loc_iou": hiou_based_errors["loc_iou"].round(num_bits),
            "loc_s2m_err": hiou_based_errors["loc_s2m_err"].round(num_bits),
            "loc_m2s_err": hiou_based_errors["loc_m2s_err"].round(num_bits),
            "loc_itf_err": hiou_based_errors["loc_itf_err"].round(num_bits),
            "loc_pcp_err": hiou_based_errors["loc_pcp_err"].round(num_bits),
            # pr curves
            "pre": pixel_level_metrics["precision"]["dynamic"].round(num_bits),
            "rec": pixel_level_metrics["recall"]["dynamic"].round(num_bits),
            # roc curves
            "tpr": pixel_level_metrics["TPR"]["dynamic"].round(num_bits),
            "fpr": pixel_level_metrics["FPR"]["dynamic"].round(num_bits),
        }

    def show(self, num_bits=4, return_ndarray: bool = False) -> dict:
        results = self.get_all_results(num_bits)
        if not return_ndarray:
            results = ndarray_to_basetype(results)
        return results


def cal_sample_wise_metrics():
    data_root = "./test_data"
    mask_paths = []
    pred_paths = []
    for file_name in os.listdir(data_root):
        file_path = os.path.join(data_root, file_name)
        if file_name.endswith("-pred.png"):
            pred_paths.append(file_path)
        else:
            mask_paths.append(file_path)

    for mask_path, pred_path in zip(mask_paths, pred_paths):
        metrics = IRSTDPerformanceAnalysis(num_bins=10)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        mask = mask > 127
        pred = pred / 255
        metrics.update(pred, mask, mask_path=mask_path)
        print(os.path.basename(mask_path))
        print(metrics.show(num_bits=3))
        print(metrics.show(num_bits=3))


def plot_average_metrics():
    import matplotlib.pyplot as plt

    data_root = "./test_data"
    mask_paths = []
    pred_paths = []
    for file_name in os.listdir(data_root):
        file_path = os.path.join(data_root, file_name)
        if file_name.endswith("-pred.png"):
            pred_paths.append(file_path)
        else:
            mask_paths.append(file_path)

    num_bins = 10
    metrics = IRSTDPerformanceAnalysis(num_bins=num_bins)
    for mask_path, pred_path in zip(mask_paths, pred_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        mask = mask > 127
        pred = pred / 255
        metrics.update(pred, mask, mask_path=mask_path)
    results = metrics.show(num_bits=3, return_ndarray=True)

    thresholds = np.linspace(0, 1, num_bins, endpoint=False)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    axes[0].plot(thresholds, results["pd_distonly"], label="pd_distonly")
    axes[0].plot(thresholds, results["fa_distonly"], label="fa_distonly")
    axes[0].legend()

    axes[1].plot(thresholds, results["pd_opdc"], label="pd_opdc")
    axes[1].plot(thresholds, results["fa_opdc"], label="fa_opdc")
    axes[1].legend()

    axes[2].plot(thresholds, results["hiou_opdc"], label="hiou_opdc")
    axes[2].plot(thresholds, results["seg_iou"], label="seg_iou")
    axes[2].plot(thresholds, results["seg_mrg_err"], label="seg_mrg_err")
    axes[2].plot(thresholds, results["seg_itf_err"], label="seg_itf_err")
    axes[2].plot(thresholds, results["seg_pcp_err"], label="seg_pcp_err")
    axes[2].legend()

    axes[3].plot(thresholds, results["loc_iou"], label="loc_iou")
    axes[3].plot(thresholds, results["loc_s2m_err"], label="loc_s2m_err")
    axes[3].plot(thresholds, results["loc_m2s_err"], label="loc_m2s_err")
    axes[3].plot(thresholds, results["loc_itf_err"], label="loc_itf_err")
    axes[3].plot(thresholds, results["loc_pcp_err"], label="loc_pcp_err")
    axes[3].legend()

    axes[4].plot(results["rec"], results["pre"], label="PR Curves")
    axes[4].plot(results["fpr"], results["tpr"], label="ROC Curves")
    axes[4].legend()
    plt.show()


if __name__ == "__main__":
    # cal_sample_wise_metrics()
    plot_average_metrics()
