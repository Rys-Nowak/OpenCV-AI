import pip
import sys

for module in ['scipy', 'scikit-image', 'opencv-python', 'numpy', "surface-distance"]:
    if module not in sys.modules:
        pip.main(['install', module])

import logging
from typing import Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode

import numpy as np
import cv2
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import filters
import surface_distance

#
# LungsSegmentation
#


class LungsSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("LungsSegmentation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ryszard Nowak (AGH)", "Eryk Mikołajek (AGH)"]
        self.parent.helpText = _("")
        self.parent.acknowledgementText = _("")


#
# LungsSegmentationParameterNode
#


@parameterNodeWrapper
class LungsSegmentationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    referenceVolume
    """

    inputVolume: vtkMRMLScalarVolumeNode
    gtVolume: vtkMRMLScalarVolumeNode
    referenceVolume: vtkMRMLSegmentationNode


#
# LungsSegmentationWidget
#


class LungsSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/LungsSegmentation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LungsSegmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        if not self._parameterNode.referenceVolume:
            firstSegNode = slicer.mrmlScene.GetFirstNodeByName("vtkMRMLSegmentationNode")
            if firstSegNode:
                self._parameterNode.referenceVolume = firstSegNode

    def setParameterNode(self, inputParameterNode: Optional[LungsSegmentationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.gtSelector.currentNode(), self.ui.referenceSelector.currentNode())


#
# LungsSegmentationLogic
#


class LungsSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return LungsSegmentationParameterNode(super().getParameterNode())
    
    def reconstruct(self, marker: np.ndarray, mask: np.ndarray, radius: int = 1):
        kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
        while True:
            expanded = cv2.dilate(src=marker, kernel=kernel)
            cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)
            if (marker == expanded).all():
                return expanded
            
            marker = expanded

    def binarize(self, lungs, max_val):
        lungs_bin_inv = np.zeros(lungs.T.shape)
        for i, dim in enumerate(lungs.T):
            _, lungs_bin_inv[i] = cv2.threshold(dim, -320, max_val, cv2.THRESH_BINARY_INV)

        lungs_bin_inv = lungs_bin_inv.T
        return lungs_bin_inv

    def predict_bodymask(self, lungs, max_val):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bodymask_pred = np.zeros(lungs.T.shape)
        for i, dim in enumerate(lungs.T):
            _, bin = cv2.threshold(dim, -200, max_val, cv2.THRESH_BINARY)
            img = cv2.erode(bin, kernel, iterations=1)
            img = cv2.dilate(img, kernel, iterations=1)
            img_neg = np.logical_not(img).astype(np.uint8)
            border_marker = np.zeros(img.shape)
            border_marker[0] = 1
            border_marker[-1] = 1
            border_marker[:, 0] = 1
            border_marker[:, -1] = 1
            border_marker = np.logical_and(img_neg, border_marker).astype(np.uint8)
            reconstructed = self.reconstruct(border_marker, img_neg)
            cleared_border = img_neg - reconstructed
            filled_img = np.logical_or(cleared_border, img).astype(np.uint8)
            bodymask_pred[i] = filled_img

        bodymask_pred = bodymask_pred.T.astype(bool)
        return self.clear_bubbles(bodymask_pred).astype(bool)

    def perform_morphological(self, lungs_air_within_body):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lungs_morphed = np.zeros(lungs_air_within_body.T.shape)
        for i, dim in enumerate(lungs_air_within_body.T):
            lungs_morphed[i] = cv2.medianBlur(dim, 7)

        lungs_morphed = lungs_morphed.T
        lungs_morphed = cv2.dilate(lungs_morphed, kernel, None, iterations=2)
        lungs_morphed = cv2.erode(lungs_morphed, kernel, None, iterations=2)
        lungs_morphed = cv2.erode(lungs_morphed, kernel, None, iterations=2)
        lungs_morphed = cv2.dilate(lungs_morphed, kernel, None, iterations=2)
        return lungs_morphed

    def clear_bubbles(self, lungs_morphed):
        lungs_processed = np.zeros(lungs_morphed.T.shape)
        for i, dim in enumerate(lungs_morphed.T):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dim.astype(np.uint8))
            for j in range(num_labels):
                if stats[j][cv2.CC_STAT_AREA] < 500:
                    labels[labels == j] = 0

            lungs_processed[i] = labels

        for i, dim in enumerate(lungs_processed):
            _, lungs_processed[i] = cv2.threshold(dim, 0, 255, cv2.THRESH_BINARY)

        lungs_processed = lungs_processed.T.astype(np.uint8)
        return lungs_processed

    def create_markers_and_gradients(self, lungs_processed, lungs):
        boundaries = np.zeros(lungs_processed.T.shape)
        gradients = np.zeros(lungs_processed.T.shape)
        sure_fg_3D = np.zeros(lungs_processed.T.shape)
        for i, (processed_img, org_img) in enumerate(zip(lungs_processed.T, lungs.T)):
            gradients[i] = filters.sobel(org_img)
            distance = ndi.distance_transform_edt(processed_img)
            _, sure_fg = cv2.threshold(distance, 0.4*distance.max(), 255, 0)
            sure_fg = sure_fg.astype(np.uint8)
            # sure_fg = cv2.erode(processed_img, np.ones((15, 15)), iterations=3).astype(np.uint8)
            sure_bg = cv2.dilate(processed_img, np.ones((7, 7)), iterations=3).astype(np.uint8)
            boundaries[i] = cv2.subtract(sure_bg, sure_fg)
            sure_fg_3D[i] = sure_fg

        boundaries = boundaries.T
        sure_fg_3D = sure_fg_3D.T
        markers, num_labels = ndi.label(sure_fg_3D)
        markers += 1
        markers[boundaries == 255] = 0
        labels_sums = np.zeros(num_labels)
        for i in range(num_labels):
            labels_sums[i] = np.sum((markers==i)*1)

        top_arg =  np.argmax(labels_sums)
        trimmed_labels = np.zeros(markers.shape)
        trimmed_labels += (markers == top_arg) * 1 # background
        for i in range(1, 4):
            labels_sums[top_arg] = 0
            top_arg = np.argmax(labels_sums)
            if top_arg != 0: # not border marker
                trimmed_labels += (markers == top_arg) * i

        markers = trimmed_labels.astype(np.int32)
        gradients /= np.max(gradients)
        gradients *= 255
        gradients = gradients.T.astype(np.int32)
        return gradients,markers

    def get_lungs_masks(self, lungs_test, lungs_done):
        left_lung_index = -1
        for slice in lungs_done:
            if np.isin(2, slice) and np.isin(3, slice):
                for row in slice:
                    if np.isin(2, row):
                        left_lung_index = 2
                        break
                    elif np.isin(3, row):
                        left_lung_index = 3
                        break
            
            if left_lung_index != -1:
                break
        
        assert left_lung_index == 2 or left_lung_index == 3
        left_lung_pred = lungs_done == left_lung_index
        right_lung_pred = lungs_done == (3 if left_lung_index == 2 else 2)
        left_lung_gt = lungs_test == 2
        right_lung_gt = lungs_test == 3
        return left_lung_pred,right_lung_pred,left_lung_gt,right_lung_gt

    def predict_lungs(self, lungs):
        max_val = np.max(lungs)
        lungs_bin_inv = self.binarize(lungs, max_val)
        bodymask_pred = self.predict_bodymask(lungs, max_val)
        lungs_air_within_body = np.logical_and(lungs_bin_inv, bodymask_pred).astype(np.uint8)
        lungs_morphed = self.perform_morphological(lungs_air_within_body)
        lungs_processed = self.clear_bubbles(lungs_morphed)
        gradients, markers = self.create_markers_and_gradients(lungs_processed, lungs)
        return watershed(gradients, markers)
    
    def calculate_dice(self, pred, ref):
        return surface_distance.compute_dice_coefficient(ref, pred)

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                gtVolume: vtkMRMLScalarVolumeNode,
                referenceVolume: vtkMRMLSegmentationNode) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be segmented
        :param referenceVolume
        """

        if not inputVolume or not referenceVolume or not gtVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()

        logging.info("Processing started")
        lungs_gt = slicer.util.arrayFromVolume(gtVolume).T.astype(bool).astype(np.uint8)
        segmentIds = [
            referenceVolume.GetSegmentation().GetSegmentIdBySegmentName("superior lobe of left lung"),
            referenceVolume.GetSegmentation().GetSegmentIdBySegmentName("inferior lobe of left lung"),
            referenceVolume.GetSegmentation().GetSegmentIdBySegmentName("superior lobe of right lung"),
            referenceVolume.GetSegmentation().GetSegmentIdBySegmentName("middle lobe of right lung"),
            referenceVolume.GetSegmentation().GetSegmentIdBySegmentName("inferior lobe of right lung")
        ]
        segmentsArrays = [slicer.util.arrayFromSegment(referenceVolume, id).T for id in segmentIds]
        lungs_total_seg = np.logical_or.reduce(segmentsArrays).astype(np.uint8)
        slicer.util.addVolumeFromArray(lungs_total_seg.T, name="Reference lungs binary")
        lungs_input = slicer.util.arrayFromVolume(inputVolume).T
        lungs_pred = self.predict_lungs(lungs_input)
        lungs_pred = (lungs_pred != 1).astype(np.uint8)
        slicer.util.addVolumeFromArray(lungs_pred.T, name="Segmented lungs")
        dice_total_seg = self.calculate_dice(lungs_total_seg, lungs_gt)
        dice_pred = self.calculate_dice(lungs_pred, lungs_gt)
        logging.info(f"Dice coefficient for reference segmentator: {dice_total_seg}")
        logging.info(f"Dice coefficient for our model: {dice_pred}")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
