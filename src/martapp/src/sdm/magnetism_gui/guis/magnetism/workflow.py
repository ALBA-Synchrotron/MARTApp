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

import sys, os
import pkg_resources

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

from sdm.magnetism_gui.guis.magnetism.ui_workflow import Ui_MagnetismMISTRALGUI
from sdm.magnetism_gui.guis.magnetism_preprocessing.magnetism_GUI import magnetism_GUI
from sdm.magnetism_gui.guis.magnetism_xmcd.magnetism_xmcd_GUI import magnetism_xmcd_GUI
from sdm.magnetism_gui.guis.magnetism_2Dreconstruction.magnetism_2Dreconstruction_GUI import magnetism_2Dreconstruction_GUI
from sdm.magnetism_gui.guis.magnetism_3Dreconstruction.absorption_3Dreconstruction_GUI import absorption_3Dreconstruction_GUI
from sdm.magnetism_gui.guis.magnetism_3Dreconstruction.signal_3Dreconstruction_GUI import signal_3Dreconstruction_GUI
from sdm.magnetism_gui.guis.manual_alignment.ManualImageAlignment import ManualImageAlignment

      
class WorkflowUI(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MagnetismMISTRALGUI()
        self.ui.setupUi(self)
        
        # Include ALBA logos
        logo_path = pkg_resources.resource_filename("sdm.magnetism_gui", "resources/ALBA_positiu.ico")
        self.ui.ALBALogo_label.setPixmap(QPixmap(logo_path))
        self.setWindowIcon(QIcon(logo_path))


        # Connections
        self.ui.selectPathTS1_pushButton.clicked.connect(lambda: self.select_folder_dialog(self.ui.pathTS1_lineEdit))
        self.ui.selectPathTS2_pushButton.clicked.connect(lambda: self.select_folder_dialog(self.ui.pathTS2_lineEdit))
        self.ui.magnetismTS1_pushButton.clicked.connect(lambda: self.magnetism(self.ui.pathTS1_lineEdit))
        self.ui.magnetismTS2_pushButton.clicked.connect(lambda: self.magnetism(self.ui.pathTS2_lineEdit))
        self.ui.magnetismXMCDTS1_pushButton.clicked.connect(lambda: self.magnetism_xmcd(self.ui.pathTS1_lineEdit))
        self.ui.magnetismXMCDTS2_pushButton.clicked.connect(lambda: self.magnetism_xmcd(self.ui.pathTS2_lineEdit))
        self.ui.ManualAliTS1_pushButton.clicked.connect(lambda: self.manualAlignment(self.ui.pathTS1_lineEdit))
        self.ui.ManualAliTS2_pushButton.clicked.connect(lambda: self.manualAlignment(self.ui.pathTS2_lineEdit))
        self.ui.magnetism2Dreconstruction_pushButton.clicked.connect(self.magnetism_2Dreconstruction)
        self.ui.absorption3Dreconstruction_pushButton.clicked.connect(self.absorption_3Dreconstruction)
        self.ui.magnetism3Dreconstruction_pushButton.clicked.connect(self.signal_3Dreconstruction)


    def magnetism(self,wg_path):
        '''
        Function to open magnetism widget. It writes the path in the widget
        '''
        wg = magnetism_GUI()
        wg.ui.path_lineEdit.setText(wg_path.text())
        wg.show()
        wg.finished.connect(lambda: self.magnetism_xmcd(wg_path)) # When window is closed, execute magnetism_xmcd pipeline


    def magnetism_xmcd(self,wg_path):
        wg = magnetism_xmcd_GUI()
        # Read if there is any file hdf5 in the path and write it in the widget.
        path = wg_path.text()
        wg.ui.path_lineEdit.setText(path)
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_stack.hdf5")]
            if stack:
                stack = sorted(stack,reverse=True)
                wg.ui.stack1_lineEdit.setText(os.path.join(path,stack[0]))
                if len(stack)>1:
                    wg.ui.stack2_lineEdit.setText(os.path.join(path,stack[1]))

        wg.show()


    def magnetism_2Dreconstruction(self):
        wg = magnetism_2Dreconstruction_GUI()
        path = self.ui.pathTS1_lineEdit.text()
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_xmcd.hdf5")]
            if stack:
                wg.ui.TS1_lineEdit.setText(os.path.join(path,stack[0]))
        path = self.ui.pathTS2_lineEdit.text()
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_xmcd.hdf5")]
            if stack:
                wg.ui.TS2_lineEdit.setText(os.path.join(path,stack[0]))

        wg.show()

    def absorption_3Dreconstruction(self):
        wg = absorption_3Dreconstruction_GUI()
        path = self.ui.pathTS1_lineEdit.text()
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_xmcd.hdf5")]
            if stack:
                wg.ui.fileHDF5Tomo_lineEdit.setText(os.path.join(path,stack[0]))
                wg.ui.outputPathTomo_lineEdit.setText(path)
                wg.ui.fileHDF5Tomo_lineEdit.editingFinished.emit()
        path = self.ui.pathTS2_lineEdit.text()
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_xmcd.hdf5")]
            if stack:
                wg.ui.fileHDF5TomoRot_lineEdit.setText(os.path.join(path,stack[0]))
                wg.ui.fileHDF5TomoRot_lineEdit.editingFinished.emit()
                wg.ui.outputPathTomoRot_lineEdit.setText(path)

        wg.show()

    def signal_3Dreconstruction(self):
        wg = signal_3Dreconstruction_GUI()
        path = self.ui.pathTS1_lineEdit.text()
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_xmcd.hdf5")]
            if stack:
                wg.ui.fileHDF5Tomo_lineEdit.setText(os.path.join(path,stack[0]))
                wg.ui.fileHDF5Tomo_lineEdit.editingFinished.emit()
                wg.ui.outputPathTomo_lineEdit.setText(path)
        path = self.ui.pathTS2_lineEdit.text()
        if os.path.exists(path):
            stack = [file for file in os.listdir(path) if file.endswith("_xmcd.hdf5")]
            if stack:
                wg.ui.fileHDF5TomoRot_lineEdit.setText(os.path.join(path,stack[0]))
                wg.ui.fileHDF5TomoRot_lineEdit.editingFinished.emit()

        wg.show()

    def manualAlignment(self,lineEdit):
        wg = ManualImageAlignment()
        path_TS = lineEdit.text()
        print(path_TS)
        if os.path.exists(str(path_TS)):
            file = [file for file in os.listdir(path_TS) if file.endswith("_xmcd.hdf5")]
            if file:
                if os.path.exists(file[0]):
                    wg.ui.fileFixed.setText(file[0])
                    wg.ui.fileMoving.setText(file[0])
                    wg.add_dset_to_wg_fixed(file[0])
                    wg.add_dset_to_wg_moving(file[0])
                    wg.ui.dsetFixed.setCurrentText("2DAlignedPositiveStack")
                    wg.ui.dsetMoving.setCurrentText("2DAlignedNegativeStack")
        wg.show()

    def select_folder_dialog(self,wg):
        n = QFileDialog().getExistingDirectory(None,"Select Path","",QFileDialog.ShowDirsOnly)
        if n!= None:
            wg.setText("%s" % n)
            return n
        else:
            return None

def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    program = WorkflowUI()
    program.show()
    sys.exit(app.exec_())
      

if __name__ == "__main__":
   main()