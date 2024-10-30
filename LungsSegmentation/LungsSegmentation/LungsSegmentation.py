import pip
import sys

for module in ['scipy', 'scikit-image', 'opencv-python', 'numpy']:
    if module in sys.modules:
        pip.main(['install', module])

import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import numpy as np
import cv2
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import filters

#
# LungsSegmentation
#


class LungsSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("LungsSegmentation")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Ryszard Nowak (AGH)", "Eryk Mikołajek (AGH)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#LungsSegmentation">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # LungsSegmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="LungsSegmentation",
        sampleName="LungsSegmentation1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "LungsSegmentation1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="LungsSegmentation1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="LungsSegmentation1",
    )

    # LungsSegmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="LungsSegmentation",
        sampleName="LungsSegmentation2",
        thumbnailFileName=os.path.join(iconsPath, "LungsSegmentation2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="LungsSegmentation2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="LungsSegmentation2",
    )


#
# LungsSegmentationParameterNode
#


@parameterNodeWrapper
class LungsSegmentationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


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
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


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

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()

        logging.info("Processing started")
        lungs_input = slicer.util.arrayFromVolume(inputVolume).T
        lungs_pred = self.predict_lungs(lungs_input)
        slicer.util.addVolumeFromArray(lungs_pred.T, name="Segmented lungs")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# LungsSegmentationTest
#


class LungsSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_LungsSegmentation1()

    def test_LungsSegmentation1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("LungsSegmentation1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = LungsSegmentationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
