"""Base Inferencer for Torch and OpenVINO."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, cast

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from torch import Tensor

from anomalib.data.utils import read_image
from anomalib.post_processing import compute_mask, superimpose_anomaly_map
from anomalib.post_processing.normalization.cdf import normalize as normalize_cdf
from anomalib.post_processing.normalization.cdf import standardize
from anomalib.post_processing.normalization.min_max import (
    normalize as normalize_min_max,
)


class Inferencer(ABC):
    """Abstract class for the inference.

    This is used by both Torch and OpenVINO inference.
    """

    @abstractmethod
    def load_model(self, path: Union[str, Path]):
        """Load Model."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, image: np.ndarray) -> Union[np.ndarray, Tensor]:
        """Pre-process."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        """Forward-Pass input to model."""
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self, predictions: Union[np.ndarray, Tensor], meta_data: Optional[Dict]
    ) -> Tuple[np.ndarray, float]:
        """Post-Process."""
        raise NotImplementedError

    def predict(
        self,
        image: Union[str, np.ndarray, Path],
        superimpose: bool = True,
        meta_data: Optional[dict] = None,
        overlay_mask: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            superimpose (bool): If this is set to True, output predictions
                will be superimposed onto the original image. If false, `predict`
                method will return the raw heatmap.

            overlay_mask (bool): If this is set to True, output segmentation mask on top of image.

        Returns:
            np.ndarray: Output predictions to be visualized.
        """
        if meta_data is None:
            if hasattr(self, "meta_data"):
                meta_data = getattr(self, "meta_data")
            else:
                meta_data = {}
        if isinstance(image, (str, Path)):
            image_arr: np.ndarray = read_image(image)
        else:  # image is already a numpy array. Kept for mypy compatibility.
            image_arr = image
        meta_data["image_shape"] = image_arr.shape[:2]

        processed_image = self.pre_process(image_arr)
        predictions = self.forward(processed_image)
        anomaly_map, pred_scores = self.post_process(predictions, meta_data=meta_data)

        # Overlay segmentation mask using raw predictions
        if overlay_mask and meta_data is not None:
            image_arr = self._superimpose_segmentation_mask(meta_data, anomaly_map, image_arr)

        if superimpose is True:
            anomaly_map = superimpose_anomaly_map(anomaly_map, image_arr)

        return anomaly_map, pred_scores

    def _superimpose_segmentation_mask(self, meta_data: dict, anomaly_map: np.ndarray, image: np.ndarray):
        """Superimpose segmentation mask on top of image.

        Args:
            meta_data (dict): Metadata of the image which contains the image size.
            anomaly_map (np.ndarray): Anomaly map which is used to extract segmentation mask.
            image (np.ndarray): Image on which segmentation mask is to be superimposed.

        Returns:
            np.ndarray: Image with segmentation mask superimposed.
        """
        pred_mask = compute_mask(anomaly_map, 0.5)  # assumes predictions are normalized.
        image_height = meta_data["image_shape"][0]
        image_width = meta_data["image_shape"][1]
        pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        boundaries = find_boundaries(pred_mask)
        outlines = dilation(boundaries, np.ones((7, 7)))
        image[outlines] = [255, 0, 0]
        return image

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            np.ndarray: Output predictions to be visualized
        """
        return self.predict(image)

    def _normalize(
        self,
        anomaly_maps: Union[Tensor, np.ndarray],
        pred_scores: Union[Tensor, np.float32],
        meta_data: Union[Dict, DictConfig],
    ) -> Tuple[Union[np.ndarray, Tensor], float]:
        """Applies normalization and resizes the image.

        Args:
            anomaly_maps (Union[Tensor, np.ndarray]): Predicted raw anomaly map.
            pred_scores (Union[Tensor, np.float32]): Predicted anomaly score
            meta_data (Dict): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.

        Returns:
            Tuple[Union[np.ndarray, Tensor], float]: Post processed predictions that are ready to be visualized and
                predicted scores.


        """

        # min max normalization
        if "min" in meta_data and "max" in meta_data:
            anomaly_maps = normalize_min_max(
                anomaly_maps, meta_data["pixel_threshold"], meta_data["min"], meta_data["max"]
            )
            pred_scores = normalize_min_max(
                pred_scores, meta_data["image_threshold"], meta_data["min"], meta_data["max"]
            )

        # standardize pixel scores
        if "pixel_mean" in meta_data.keys() and "pixel_std" in meta_data.keys():
            anomaly_maps = standardize(
                anomaly_maps, meta_data["pixel_mean"], meta_data["pixel_std"], center_at=meta_data["image_mean"]
            )
            anomaly_maps = normalize_cdf(anomaly_maps, meta_data["pixel_threshold"])

        # standardize image scores
        if "image_mean" in meta_data.keys() and "image_std" in meta_data.keys():
            pred_scores = standardize(pred_scores, meta_data["image_mean"], meta_data["image_std"])
            pred_scores = normalize_cdf(pred_scores, meta_data["image_threshold"])

        return anomaly_maps, float(pred_scores)

    def _load_meta_data(
        self, path: Optional[Union[str, Path]] = None
    ) -> Union[DictConfig, Dict[str, Union[float, np.ndarray, Tensor]]]:
        """Loads the meta data from the given path.

        Args:
            path (Optional[Union[str, Path]], optional): Path to JSON file containing the metadata.
                If no path is provided, it returns an empty dict. Defaults to None.

        Returns:
            Union[DictConfig, Dict]: Dictionary containing the metadata.
        """
        meta_data: Union[DictConfig, Dict[str, Union[float, np.ndarray, Tensor]]] = {}
        if path is not None:
            config = OmegaConf.load(path)
            meta_data = cast(DictConfig, config)
        return meta_data
