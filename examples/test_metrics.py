import os
import sys
import unittest
from pprint import pprint

import cv2

sys.path.append("..")
import py_irstd_metrics

default_results = {
    "v0_1_0": {
        "basic_fa": 0.000677490234375,
        "basic_pd": 0.4,
        "dist_fa": 0.000677490234375,
        "dist_pd": 0.4,
        "f1": 0.7234042553191489,
        "fpr": 0.111410725118057,
        "iou": 0.5666666666666667,
        "loc_iou": 0.42857142857142855,
        "loc_itf_err": 0.2857142857142857,
        "loc_m2s_err": 0.0,
        "loc_pcp_err": 0.0,
        "loc_s2m_err": 0.2857142857142857,
        "niou": 0.5435835351089588,
        "opdc_fa": 4.8828125e-05,
        "opdc_hiou": 0.22753395044416713,
        "opdc_pd": 0.6,
        "pre": 0.5202976412128605,
        "rec": 0.8506172839506172,
        "seg_iou": 0.53091255103639,
        "seg_itf_err": 0.1818054062636106,
        "seg_mrg_err": 0.19258553623878702,
        "seg_pcp_err": 0.09469650646121235,
        "tpr": 0.8506172839506172,
    },
}


class CheckMetricTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fmt: off
        irstd_cm_metrics = py_irstd_metrics.CMMetrics(
            num_bins=10,
            threshold=0.5,
            metric_handlers={
                # values
                "iou": py_irstd_metrics.IoUHandler(with_dynamic=False, with_binary=True, sample_based=False),
                "normalized_iou": py_irstd_metrics.IoUHandler(with_dynamic=False, with_binary=True, sample_based=True),
                "f1": py_irstd_metrics.FmeasureHandler(with_dynamic=False, with_binary=True, sample_based=False, beta=1),
                # curves
                "pre": py_irstd_metrics.PrecisionHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "rec": py_irstd_metrics.RecallHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "tpr": py_irstd_metrics.TPRHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "fpr": py_irstd_metrics.FPRHandler(with_dynamic=True, with_binary=False, sample_based=False),
            },
        )

        basic_irstd_metrics = py_irstd_metrics.ProbabilityDetectionAndFalseAlarmRate(num_bins=8, distance_threshold=3)
        opdc_irstd_metrics = py_irstd_metrics.MatchingBasedMetrics(num_bins=8, matching_method=py_irstd_metrics.OPDCMatching(overlap_threshold=0.5, distance_threshold=3))
        dist_irstd_metrics = py_irstd_metrics.MatchingBasedMetrics(num_bins=8, matching_method=py_irstd_metrics.DistanceOnlyMatching(distance_threshold=3))
        irstd_error_analysis = py_irstd_metrics.HierarchicalIoUBasedErrorAnalysis(num_bins=8, overlap_threshold=0.5, distance_threshold=3)
        # fmt: on

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
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            mask = mask > 127
            pred = pred / 255

            irstd_cm_metrics.update(pred, mask)

            basic_irstd_metrics.update(pred, mask)
            opdc_irstd_metrics.update(pred, mask)
            dist_irstd_metrics.update(pred, mask)
            irstd_error_analysis.update(pred, mask)

        cm_metrics = irstd_cm_metrics.get()
        basic_metrics = basic_irstd_metrics.get()
        opdc_metrics = opdc_irstd_metrics.get()
        dist_metrics = dist_irstd_metrics.get()
        error_analysis = irstd_error_analysis.get()

        cls.curr_results = {
            "iou": cm_metrics["iou"]["binary"].item(),
            "niou": cm_metrics["normalized_iou"]["binary"].item(),
            "f1": cm_metrics["f1"]["binary"].item(),
            "pre": cm_metrics["pre"]["dynamic"].mean().item(),
            "rec": cm_metrics["rec"]["dynamic"].mean().item(),
            "tpr": cm_metrics["tpr"]["dynamic"].mean().item(),
            "fpr": cm_metrics["fpr"]["dynamic"].mean().item(),
            #
            "basic_pd": basic_metrics["probability_detection"][4].item(),
            "basic_fa": basic_metrics["false_alarm"][4].item(),
            #
            "dist_fa": dist_metrics["false_alarm"][4].item(),
            "dist_pd": dist_metrics["probability_detection"][4].item(),
            #
            "opdc_fa": opdc_metrics["false_alarm"][4].item(),
            "opdc_pd": opdc_metrics["probability_detection"][4].item(),
            "opdc_hiou": opdc_metrics["hiou"][4].item(),
            #
            "seg_iou": error_analysis["seg_iou"][4].item(),
            "seg_mrg_err": error_analysis["seg_mrg_err"][4].item(),
            "seg_itf_err": error_analysis["seg_itf_err"][4].item(),
            "seg_pcp_err": error_analysis["seg_pcp_err"][4].item(),
            "loc_iou": error_analysis["loc_iou"][4].item(),
            "loc_s2m_err": error_analysis["loc_s2m_err"][4].item(),
            "loc_m2s_err": error_analysis["loc_m2s_err"][4].item(),
            "loc_itf_err": error_analysis["loc_itf_err"][4].item(),
            "loc_pcp_err": error_analysis["loc_pcp_err"][4].item(),
        }

        print("Current results:")
        pprint(cls.curr_results)
        cls.default_results = default_results["v0_1_0"]

    def test_pd_fa(self):
        self.assertEqual(self.curr_results["basic_pd"], self.default_results["dist_pd"])
        self.assertEqual(self.curr_results["basic_fa"], self.default_results["dist_fa"])

    def test_metrics(self):
        self.assertEqual(self.curr_results["basic_pd"], self.default_results["basic_pd"])
        self.assertEqual(self.curr_results["basic_fa"], self.default_results["basic_fa"])

        self.assertEqual(self.curr_results["dist_fa"], self.default_results["dist_fa"])
        self.assertEqual(self.curr_results["dist_pd"], self.default_results["dist_pd"])

        self.assertEqual(self.curr_results["opdc_fa"], self.default_results["opdc_fa"])
        self.assertEqual(self.curr_results["opdc_pd"], self.default_results["opdc_pd"])
        self.assertEqual(self.curr_results["opdc_hiou"], self.default_results["opdc_hiou"])

        self.assertEqual(self.curr_results["seg_iou"], self.default_results["seg_iou"])
        self.assertEqual(self.curr_results["seg_mrg_err"], self.default_results["seg_mrg_err"])
        self.assertEqual(self.curr_results["seg_itf_err"], self.default_results["seg_itf_err"])
        self.assertEqual(self.curr_results["seg_pcp_err"], self.default_results["seg_pcp_err"])
        self.assertEqual(self.curr_results["loc_iou"], self.default_results["loc_iou"])
        self.assertEqual(self.curr_results["loc_s2m_err"], self.default_results["loc_s2m_err"])
        self.assertEqual(self.curr_results["loc_m2s_err"], self.default_results["loc_m2s_err"])
        self.assertEqual(self.curr_results["loc_itf_err"], self.default_results["loc_itf_err"])
        self.assertEqual(self.curr_results["loc_pcp_err"], self.default_results["loc_pcp_err"])


if __name__ == "__main__":
    unittest.main()
