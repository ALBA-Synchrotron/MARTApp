# -*- coding: utf-8 -*-
# Copyright (C) 2014-2025 ALBA Synchrotron
#
# Authors: A. Estela Herguedas Alonso, Joaquin Gomez Sanchez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_magnetismXMCD(object):
    def setupUi(self, magnetismXMCD):
        magnetismXMCD.setObjectName("magnetismXMCD")
        magnetismXMCD.resize(780, 812)
        magnetismXMCD.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("magnetism_xmcd\\ALBA_positiu.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        magnetismXMCD.setWindowIcon(icon)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(magnetismXMCD)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_8 = QtWidgets.QLabel(magnetismXMCD)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.groupBox_7 = QtWidgets.QGroupBox(magnetismXMCD)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_9 = QtWidgets.QLabel(self.groupBox_7)
        self.label_9.setObjectName("label_9")
        self.gridLayout_9.addWidget(self.label_9, 2, 0, 1, 1)
        self.selectStack1_pushButton = QtWidgets.QPushButton(self.groupBox_7)
        self.selectStack1_pushButton.setObjectName("selectStack1_pushButton")
        self.gridLayout_9.addWidget(self.selectStack1_pushButton, 1, 2, 1, 1)
        self.stack1_lineEdit = QtWidgets.QLineEdit(self.groupBox_7)
        self.stack1_lineEdit.setObjectName("stack1_lineEdit")
        self.gridLayout_9.addWidget(self.stack1_lineEdit, 1, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_7)
        self.label_10.setObjectName("label_10")
        self.gridLayout_9.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_7)
        self.label_11.setObjectName("label_11")
        self.gridLayout_9.addWidget(self.label_11, 1, 0, 1, 1)
        self.selectPath_pushButton = QtWidgets.QPushButton(self.groupBox_7)
        self.selectPath_pushButton.setObjectName("selectPath_pushButton")
        self.gridLayout_9.addWidget(self.selectPath_pushButton, 0, 2, 1, 1)
        self.path_lineEdit = QtWidgets.QLineEdit(self.groupBox_7)
        self.path_lineEdit.setObjectName("path_lineEdit")
        self.gridLayout_9.addWidget(self.path_lineEdit, 0, 1, 1, 1)
        self.stack2_lineEdit = QtWidgets.QLineEdit(self.groupBox_7)
        self.stack2_lineEdit.setObjectName("stack2_lineEdit")
        self.gridLayout_9.addWidget(self.stack2_lineEdit, 2, 1, 1, 1)
        self.selectStack2_pushButton = QtWidgets.QPushButton(self.groupBox_7)
        self.selectStack2_pushButton.setObjectName("selectStack2_pushButton")
        self.gridLayout_9.addWidget(self.selectStack2_pushButton, 2, 2, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_9, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_7)
        self.groupBox_8 = QtWidgets.QGroupBox(magnetismXMCD)
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.groupBox_8)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_12 = QtWidgets.QLabel(self.groupBox_8)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 0, 0, 1, 1)
        self.pxCutFromBorders_spinBox = QtWidgets.QSpinBox(self.groupBox_8)
        self.pxCutFromBorders_spinBox.setMaximum(99999999)
        self.pxCutFromBorders_spinBox.setObjectName("pxCutFromBorders_spinBox")
        self.gridLayout_3.addWidget(self.pxCutFromBorders_spinBox, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_8)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 0, 2, 1, 1)
        self.excludeSample_lineEdit = QtWidgets.QLineEdit(self.groupBox_8)
        self.excludeSample_lineEdit.setObjectName("excludeSample_lineEdit")
        self.gridLayout_3.addWidget(self.excludeSample_lineEdit, 0, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_8)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.saveToAlign_checkBox = QtWidgets.QCheckBox(self.groupBox_8)
        self.saveToAlign_checkBox.setText("")
        self.saveToAlign_checkBox.setObjectName("saveToAlign_checkBox")
        self.gridLayout_3.addWidget(self.saveToAlign_checkBox, 1, 1, 1, 1)
        self.horizontalLayout_11.addLayout(self.gridLayout_3)
        self.verticalLayout_2.addWidget(self.groupBox_8)
        self.groupBox_9 = QtWidgets.QGroupBox(magnetismXMCD)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox_9)
        self.groupBox_10.setObjectName("groupBox_10")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_10)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_14 = QtWidgets.QLabel(self.groupBox_10)
        self.label_14.setObjectName("label_14")
        self.gridLayout_12.addWidget(self.label_14, 2, 0, 1, 1)
        self.roi2DAli_checkBox = QtWidgets.QCheckBox(self.groupBox_10)
        self.roi2DAli_checkBox.setObjectName("roi2DAli_checkBox")
        self.gridLayout_12.addWidget(self.roi2DAli_checkBox, 0, 0, 1, 1)
        self.mask2DAli_checkBox = QtWidgets.QCheckBox(self.groupBox_10)
        self.mask2DAli_checkBox.setObjectName("mask2DAli_checkBox")
        self.gridLayout_12.addWidget(self.mask2DAli_checkBox, 0, 1, 1, 1)
        self.roiSize2DAli_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_10)
        self.roiSize2DAli_doubleSpinBox.setDecimals(3)
        self.roiSize2DAli_doubleSpinBox.setMaximum(1.0)
        self.roiSize2DAli_doubleSpinBox.setSingleStep(0.01)
        self.roiSize2DAli_doubleSpinBox.setProperty("value", 0.8)
        self.roiSize2DAli_doubleSpinBox.setObjectName("roiSize2DAli_doubleSpinBox")
        self.gridLayout_12.addWidget(self.roiSize2DAli_doubleSpinBox, 2, 1, 1, 1)
        self.subpixelAli_checkBox = QtWidgets.QCheckBox(self.groupBox_10)
        self.subpixelAli_checkBox.setObjectName("subpixelAli_checkBox")
        self.gridLayout_12.addWidget(self.subpixelAli_checkBox, 1, 0, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_12, 3, 0, 1, 1)
        self.algorithm2DAli_comboBox = QtWidgets.QComboBox(self.groupBox_10)
        self.algorithm2DAli_comboBox.setObjectName("algorithm2DAli_comboBox")
        self.algorithm2DAli_comboBox.addItem("")
        self.algorithm2DAli_comboBox.addItem("")
        self.algorithm2DAli_comboBox.addItem("")
        self.algorithm2DAli_comboBox.addItem("")
        self.algorithm2DAli_comboBox.addItem("")
        self.gridLayout_11.addWidget(self.algorithm2DAli_comboBox, 0, 0, 1, 1)
        self.gridLayout_10.addWidget(self.groupBox_10, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_9)
        self.groupBox_12 = QtWidgets.QGroupBox(magnetismXMCD)
        self.groupBox_12.setObjectName("groupBox_12")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.groupBox_12)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.croppingSize_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_12)
        self.croppingSize_doubleSpinBox.setMaximum(1.0)
        self.croppingSize_doubleSpinBox.setSingleStep(0.1)
        self.croppingSize_doubleSpinBox.setProperty("value", 0.95)
        self.croppingSize_doubleSpinBox.setObjectName("croppingSize_doubleSpinBox")
        self.gridLayout.addWidget(self.croppingSize_doubleSpinBox, 0, 1, 1, 1)
        self.croppingMethod_comboBox = QtWidgets.QComboBox(self.groupBox_12)
        self.croppingMethod_comboBox.setObjectName("croppingMethod_comboBox")
        self.croppingMethod_comboBox.addItem("")
        self.croppingMethod_comboBox.addItem("")
        self.gridLayout.addWidget(self.croppingMethod_comboBox, 0, 0, 1, 1)
        self.horizontalLayout_15.addLayout(self.gridLayout)
        self.verticalLayout_2.addWidget(self.groupBox_12)
        self.groupBox_11 = QtWidgets.QGroupBox(magnetismXMCD)
        self.groupBox_11.setObjectName("groupBox_11")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.groupBox_11)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.algorithmTiltAli_comboBox = QtWidgets.QComboBox(self.groupBox_11)
        self.algorithmTiltAli_comboBox.setObjectName("algorithmTiltAli_comboBox")
        self.algorithmTiltAli_comboBox.addItem("")
        self.algorithmTiltAli_comboBox.addItem("")
        self.algorithmTiltAli_comboBox.addItem("")
        self.algorithmTiltAli_comboBox.addItem("")
        self.gridLayout_2.addWidget(self.algorithmTiltAli_comboBox, 0, 0, 1, 1)
        self.maskTiltAli_checkBox = QtWidgets.QCheckBox(self.groupBox_11)
        self.maskTiltAli_checkBox.setEnabled(False)
        self.maskTiltAli_checkBox.setObjectName("maskTiltAli_checkBox")
        self.gridLayout_2.addWidget(self.maskTiltAli_checkBox, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox_11)
        self.label.setEnabled(False)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.nfid_spinBox = QtWidgets.QSpinBox(self.groupBox_11)
        self.nfid_spinBox.setEnabled(False)
        self.nfid_spinBox.setMaximum(999999999)
        self.nfid_spinBox.setDisplayIntegerBase(30)
        self.nfid_spinBox.setObjectName("nfid_spinBox")
        self.horizontalLayout_2.addWidget(self.nfid_spinBox)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)
        self.repeatTiltAlign_checkBox = QtWidgets.QCheckBox(self.groupBox_11)
        self.repeatTiltAlign_checkBox.setEnabled(False)
        self.repeatTiltAlign_checkBox.setObjectName("repeatTiltAlign_checkBox")
        self.gridLayout_2.addWidget(self.repeatTiltAlign_checkBox, 2, 0, 1, 1)
        self.magTiltAli_checkBox = QtWidgets.QCheckBox(self.groupBox_11)
        self.magTiltAli_checkBox.setEnabled(False)
        self.magTiltAli_checkBox.setCheckable(True)
        self.magTiltAli_checkBox.setObjectName("magTiltAli_checkBox")
        self.gridLayout_2.addWidget(self.magTiltAli_checkBox, 1, 1, 1, 1)
        self.gridLayout_13.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_11)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.start_pushButton = QtWidgets.QPushButton(magnetismXMCD)
        self.start_pushButton.setObjectName("start_pushButton")
        self.horizontalLayout_9.addWidget(self.start_pushButton)
        self.visualization_pushButton = QtWidgets.QPushButton(magnetismXMCD)
        self.visualization_pushButton.setObjectName("visualization_pushButton")
        self.horizontalLayout_9.addWidget(self.visualization_pushButton)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.notes = QtWidgets.QTextBrowser(magnetismXMCD)
        self.notes.setObjectName("notes")
        self.verticalLayout_5.addWidget(self.notes)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(magnetismXMCD)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(magnetismXMCD)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.verticalLayout_5.addLayout(self.horizontalLayout)

        self.retranslateUi(magnetismXMCD)
        self.buttonBox.accepted.connect(magnetismXMCD.accept) # type: ignore
        self.buttonBox.rejected.connect(magnetismXMCD.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(magnetismXMCD)
        magnetismXMCD.setTabOrder(self.path_lineEdit, self.selectPath_pushButton)
        magnetismXMCD.setTabOrder(self.selectPath_pushButton, self.stack1_lineEdit)
        magnetismXMCD.setTabOrder(self.stack1_lineEdit, self.selectStack1_pushButton)
        magnetismXMCD.setTabOrder(self.selectStack1_pushButton, self.stack2_lineEdit)
        magnetismXMCD.setTabOrder(self.stack2_lineEdit, self.selectStack2_pushButton)
        magnetismXMCD.setTabOrder(self.selectStack2_pushButton, self.pxCutFromBorders_spinBox)
        magnetismXMCD.setTabOrder(self.pxCutFromBorders_spinBox, self.excludeSample_lineEdit)
        magnetismXMCD.setTabOrder(self.excludeSample_lineEdit, self.roiSize2DAli_doubleSpinBox)
        magnetismXMCD.setTabOrder(self.roiSize2DAli_doubleSpinBox, self.start_pushButton)
        magnetismXMCD.setTabOrder(self.start_pushButton, self.visualization_pushButton)
        magnetismXMCD.setTabOrder(self.visualization_pushButton, self.notes)

    def retranslateUi(self, magnetismXMCD):
        _translate = QtCore.QCoreApplication.translate
        magnetismXMCD.setWindowTitle(_translate("magnetismXMCD", "MARTApp"))
        self.label_8.setToolTip(_translate("magnetismXMCD", "This pipeline computes the absorbtion and the signal of two given polarization stacks previously computed using the magnetism pipeline. It offers different 2D alignment and cropping methods, and (tilt) aligns using OpticalFlow.\n"
"[exclude angles] >> [crop] >> [create mask] >> 2D alignment polarizations >> crop >> compensation >> -ln(stacks) >> pos_stack +- neg_stack >> [tilt alignment]"))
        self.label_8.setText(_translate("magnetismXMCD", "XMCD"))
        self.groupBox_7.setTitle(_translate("magnetismXMCD", "Input data"))
        self.label_9.setToolTip(_translate("magnetismXMCD", "Path of HDF5 file with the normalized stack for polarization -1."))
        self.label_9.setText(_translate("magnetismXMCD", "Stack polarization -1"))
        self.selectStack1_pushButton.setText(_translate("magnetismXMCD", "Select"))
        self.label_10.setToolTip(_translate("magnetismXMCD", "Path of the data to save"))
        self.label_10.setText(_translate("magnetismXMCD", "Path"))
        self.label_11.setToolTip(_translate("magnetismXMCD", "Path of HDF5 file with the normalized stack for polarization +1."))
        self.label_11.setText(_translate("magnetismXMCD", "Stack polarization 1"))
        self.selectPath_pushButton.setText(_translate("magnetismXMCD", "Select"))
        self.selectStack2_pushButton.setText(_translate("magnetismXMCD", "Select"))
        self.groupBox_8.setTitle(_translate("magnetismXMCD", "Initial"))
        self.label_12.setToolTip(_translate("magnetismXMCD", "Number of pixels to cut the image from every border."))
        self.label_12.setText(_translate("magnetismXMCD", "Cut pixels from borders"))
        self.label_13.setToolTip(_translate("magnetismXMCD", "List of samples images in the stack to be exlcuded. The number in the stack (not the angle) should be indicated.\n"
"Stacks starts in 0."))
        self.label_13.setText(_translate("magnetismXMCD", "Exclude angular projection"))
        self.label_2.setText(_translate("magnetismXMCD", "Save images used for alignment"))
        self.groupBox_9.setTitle(_translate("magnetismXMCD", "2D Alignment"))
        self.groupBox_10.setToolTip(_translate("magnetismXMCD", "Method used for the 2D alignment between polarizations."))
        self.groupBox_10.setTitle(_translate("magnetismXMCD", "Method"))
        self.label_14.setToolTip(_translate("magnetismXMCD", "ROI size from the center (in percentage, e.g. 0.5 or 0.8) for the alignment methods crosscorr and corrcoeff"))
        self.label_14.setText(_translate("magnetismXMCD", "ROI size from centre (%)"))
        self.roi2DAli_checkBox.setToolTip(_translate("magnetismXMCD", "If indicated, the alignment will be done with a ROI and the given shift will be used for the entire images."))
        self.roi2DAli_checkBox.setText(_translate("magnetismXMCD", "Select ROI"))
        self.mask2DAli_checkBox.setToolTip(_translate("magnetismXMCD", "If indicated, the 2D alignment between polarization (angle-angle) is going to be done using a mask (tanh of images) as intermediary."))
        self.mask2DAli_checkBox.setText(_translate("magnetismXMCD", "Use mask"))
        self.subpixelAli_checkBox.setText(_translate("magnetismXMCD", "Subpixel alignment"))
        self.algorithm2DAli_comboBox.setToolTip(_translate("magnetismXMCD", "Select algorithm for 2D alignment"))
        self.algorithm2DAli_comboBox.setItemText(0, _translate("magnetismXMCD", "pyStackReg"))
        self.algorithm2DAli_comboBox.setItemText(1, _translate("magnetismXMCD", "Cross-Correlation Fourier"))
        self.algorithm2DAli_comboBox.setItemText(2, _translate("magnetismXMCD", "Cross-Correlation"))
        self.algorithm2DAli_comboBox.setItemText(3, _translate("magnetismXMCD", "Correlation Coefficient"))
        self.algorithm2DAli_comboBox.setItemText(4, _translate("magnetismXMCD", "OpticalFlow"))
        self.groupBox_12.setToolTip(_translate("magnetismXMCD", "Choose if after the 2D alignment the image should be cropped or whether the pixels with value less than 0.1 should be filled with ones in order to create background and void problems with the natural logarithm."))
        self.groupBox_12.setTitle(_translate("magnetismXMCD", "Cropping method for borders after alignment"))
        self.croppingMethod_comboBox.setItemText(0, _translate("magnetismXMCD", "Cropping"))
        self.croppingMethod_comboBox.setItemText(1, _translate("magnetismXMCD", "Filling"))
        self.groupBox_11.setTitle(_translate("magnetismXMCD", "Tilt Alignment"))
        self.algorithmTiltAli_comboBox.setItemText(0, _translate("magnetismXMCD", "None"))
        self.algorithmTiltAli_comboBox.setItemText(1, _translate("magnetismXMCD", "pyStackReg"))
        self.algorithmTiltAli_comboBox.setItemText(2, _translate("magnetismXMCD", "CT Align"))
        self.algorithmTiltAli_comboBox.setItemText(3, _translate("magnetismXMCD", "Optical Flow"))
        self.maskTiltAli_checkBox.setToolTip(_translate("magnetismXMCD", "If indicated, the tilt alignment of the absorption and the signal stacks is going to be done using a mask (tanh of images) as intermediary."))
        self.maskTiltAli_checkBox.setText(_translate("magnetismXMCD", "Use mask"))
        self.label.setText(_translate("magnetismXMCD", "Number fiducials"))
        self.nfid_spinBox.setToolTip(_translate("magnetismXMCD", "Value of number of fiducials for tilt alignment with CT Align"))
        self.repeatTiltAlign_checkBox.setText(_translate("magnetismXMCD", "Repeat only tilt alignment"))
        self.magTiltAli_checkBox.setText(_translate("magnetismXMCD", "Align with magnetic signal"))
        self.start_pushButton.setToolTip(_translate("magnetismXMCD", "Execute pipeline \"magnetism_xmcd\""))
        self.start_pushButton.setText(_translate("magnetismXMCD", "GO!"))
        self.visualization_pushButton.setToolTip(_translate("magnetismXMCD", "Open HDF5 with Napari"))
        self.visualization_pushButton.setText(_translate("magnetismXMCD", "Visualization"))
        self.label_3.setText(_translate("magnetismXMCD", "Copyright © 2024 ALBA-CELLS"))
