Automated assessment of volume of interest (VOI).

Automated assessment of volume of interest was accomplished using deep neural network model. Additional imaging dataset from head and neck cancer CT atlas [1,2] was used as a separate training dataset. It consisted of 430 whole-body PET-CTs or abdominal CTs conducted before and after radiotherapy of 215 head and neck squamous carcinoma patients, detailed description of this dataset can be found in data descriptor [1]
Studies have demonstrated the significance of utilizing body composition measurements derived from abdominal region [3], which was the reason for establishing VOI as a region between upper part of pelvic bone, and bottom part of the lung. Volume between upper and bottom slice was manually chosen and established as volume of interest. Digital Imaging and Communications in Medicine (DICOM) files were then preprocessed using a custom script composed of two main functions – one that read the DICOM files, and the other used for image standardization, generated arrays were then saved as a nifty files using nibabel library [4]. To each axial slice a label one or zero was assigned, depending on whether the slice was within VOI or outside VOI. Afterwards Medical Open Network for Artificial Intelligence (MONAI) [5] library was used to develop a binary classifier of axial slices.  Architecture used in our classification model was DenseNet121 with cross-entropy as a loss function, adam optimizer, and ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) as a metric. The code used in this stage follows closely example at MONAI repository (https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb).
Five models were trained using five-fold cross validation. The best performing models during validation were used for classification of axial slices in the test dataset – which consisted of imaging data of NSCLC patients. In particular, each axial slice in the test set received label one if at least three of the five trained models indicated that the slice is within VOI. Otherwise, the slice received label 0. Method classification capabilities was evaluated using accuracy, precision, recall and f1-score.  Predictions were also subsequently manually checked by an experienced reader with more than four years of practice in imaging body composition assessment. To assess VOI, results of classification model (either zero or one assigned to each axial slice) were used to find two axial slices in each examination, corresponding to the upper part of the pelvic bone, and bottom part of the lung. They were found by solving some optimization problem...

[1]	Grossberg AJ, Mohamed ASR, Elhalawani H, Bennett WC, Smith KE, Nolan TS, et al. Imaging and clinical data archive for head and neck squamous cell carcinoma patients treated with radiotherapy. Sci Data 2018;5:180173. https://doi.org/10.1038/sdata.2018.173.
[2]	Grossberg A, Mohamed A, Elhalawani H, Bennett W, Smith K, Nolan T, et al. Data from Head and Neck Cancer CT Atlas. The Cancer Imaging Archive 2017. https://doi.org/10.7937/K9/TCIA.2017.umz8dv6s.
[3]	Bates DDB, Pickhardt PJ. CT-Derived Body Composition Assessment as a Prognostic Tool in Oncologic Patients: From Opportunistic Research to Artificial Intelligence–Based Clinical Implementation. American Journal of Roentgenology 2022;219:671–80. https://doi.org/10.2214/AJR.22.27749/ASSET/IMAGES/LARGE/22_27749_05_CMYK.JPEG.
[4]	Brett M, Markiewicz CJ, Hanke M, Côté M-A, Cipollini B, McCarthy P, et al. nipy/nibabel: 3.2.1. Zenodo 2020. https://doi.org/10.5281/ZENODO.4295521.
[5]	Cardoso MJ, Li W, Brown R, Ma N, Kerfoot E, Wang Y, et al. MONAI: An open-source framework for deep learning in healthcare n.d. https://doi.org/10.48550/arXiv.2211.02701.


Task:
Using the data in oznaczenia.txt file create training dataset for binary classification. For example, all axial slices in image 0001_1_.nii.gz ranging from 105 to 137 should be assigned label 1 and the remaining slices should be assigned label 0. Use these data to train a classifier (like the one at https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb). Note that the data is not balanced - use appropriate measures to address this problem. After applying the classifier to a 3D image (e.g. using majority voting as in the description above) you may get a sequence of zeros and ones like {0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0}. Design an algorithm which converts this sequence to a VOI, that is the algorithm should indicate the initial and final slice of the abdominal part of body in 3D CT. Experiment with different classifiers to select the best one.
The classificator and the algorithm would be applied to a test set. The quality measure will be the difference between ground truth positions of VOI limiting slices and predicted positions of VOI limiting slices.
