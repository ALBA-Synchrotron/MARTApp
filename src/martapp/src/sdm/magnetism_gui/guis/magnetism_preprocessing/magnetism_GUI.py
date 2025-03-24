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

import subprocess
from datetime import datetime
import os
import sys
import pkg_resources

import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QListWidget, QVBoxLayout, QPushButton, QAbstractItemView
from PyQt5.QtCore import pyqtSlot,pyqtSignal, QThread, Qt
from PyQt5.QtGui import QTextCharFormat, QColor, QIcon
import re
import h5py
import napari
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,RangeSlider

from .ui_magnetism import Ui_magnetism


class runMagnetism(QThread):
    '''
    Executes a program as a subprocess and send signals to read output when finishes.
    '''
    out_signal = pyqtSignal(str)
    err_signal = pyqtSignal(str)

    def __init__(self, command, path):
        super().__init__()
        self.command = command
        self.path = path

    def run(self):
        # try:
        #     os.chdir(self.path)
        #     print(' '.join(self.command))
        #     process = subprocess.Popen(self.command, cwd=self.path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #     if process.stdout:
        #         self.out_signal.emit(process.stdout.read().decode().strip()) # Send output as signal
        #     if process.stderr:
        #         self.err_signal.emit(process.stderr.read().decode().strip()) # Send error as signal
        #     process.wait() # Wait for process to finish
        # except Exception as e:           
        #     self.err_signal.emit(str(e))
        try:
            os.chdir(self.path)
            print(' '.join(self.command))
            process = subprocess.Popen(self.command, cwd=self.path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
            while True:
                output = process.stdout.readline().decode().strip() if process.stdout else ''
                if output:
                    self.out_signal.emit('\n' + output)

                error = process.stderr.readline().decode().strip() if process.stderr else ''
                if error:
                    self.err_signal.emit('\n' + error)
                
                if output == '' and process.poll() is not None:
                    break

            process.wait() # Wait for process to finish
        except Exception as e:
            self.err_signal.emit(str(e))

class visualization_plt(QThread):
    '''
    Opens a stack for visualization using matplotlib.
    Sliders allow to modify color range and number of slice for visualization.
    '''
    def __init__(self,stack):
        super().__init__()
        # Create fig.
        if len(np.shape(stack)) == 2: # Convert image nxm to 1xnxm 
            stack = np.reshape(stack,
                               (1,np.shape(stack)[0],np.shape(stack)[1])
                            )
        self.stack = stack
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(stack[0,:,:],
                        cmap='gray',
                        vmin=np.min(stack),
                        vmax=np.max(stack),
                        aspect='equal')
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Create slider to change idx.
        self.ax_idx = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.idx_slider = Slider(ax=self.ax_idx,
                            label='No Image',
                            valmin=0,
                            valmax=np.shape(stack)[0]-1,
                            valinit=np.shape(stack)[0]/2,
                            valstep = 1,
                            orientation='horizontal'
                            )
        self.idx_slider.on_changed(self.update_idx)

        # Create slider to change color range.
        self.ax_clim = self.fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        self.clim_slider = RangeSlider(ax=self.ax_clim,
                            label='Clim',
                            valmin=np.min(stack),
                            valmax=np.max(stack),
                            orientation='vertical'
                            )
        self.clim_slider.on_changed(self.update_clim)
        plt.show(block=False)
        plt.ioff()

    def update_idx(self,val):
        self.img.set_data(self.stack[val,:,:])
        self.fig.canvas.draw()
        # plt.pause(0.001)
        plt.show(block=False)
        plt.ioff()

    def update_clim(self,val):
        self.img.norm.vmin = val[0]
        self.img.norm.vmax = val[1]
        self.fig.canvas.draw()
        # plt.pause(0.001)
        plt.show(block=False)
        plt.ioff()
        
    def on_close(self,ev):
        plt.close('all')
        att = list(self.__dict__.keys())
        for i in att:
            delattr(self, i)
      
        
class magnetism_GUI(QDialog):
    '''
     Main pyqt5 GUI
    '''
    def __init__(self):
        '''
        Initialization
        '''
        super().__init__()
        self.ui = Ui_magnetism() # Open GUI
        self.ui.setupUi(self) # Set GUI
        
        # Include ALBA logos
        logo_path = pkg_resources.resource_filename("sdm.magnetism_gui", "resources/ALBA_positiu.ico")
        self.setWindowIcon(QIcon(logo_path))

        # Connections
        self.ui.selectPath_pushButton.clicked.connect(lambda: self.select_folder_dialog(self.ui.path_lineEdit)) # Open dialog to select folder.
        self.ui.selectTxt_pushButton.clicked.connect(lambda: self.select_file_dialog(self.ui.selectTxt_lineEdit,'Text Files (*.txt)')) # Open dialog to select .txt.
        self.ui.selectDB_pushButton.clicked.connect(lambda: self.select_file_dialog(self.ui.selectDB_lineEdit,'JSON Source File (*.json)')) # Open dialog to select .json
        self.ui.start_pushButton.clicked.connect(self.start_program) # When clicking "GO!" execute pipeline.
        self.ui.visualization_pushButton.clicked.connect(self.visualization) # When clicking "Visualization" open dialog to select HDF5 for visualization.
        
        self.thread = None

    @pyqtSlot()
    def start_program(self):
        '''
        Function to read the arguments and execute the magnetism pipeline.
        '''
        self.ui.notes.clear()  # Clear the text edit widget
        t = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.ui.notes.append(t+"\tRunning...")
        self.ui.start_pushButton.setEnabled(False)  # Disable the start button
        # Execute the magnetism program as a command-line program
        arguments,path = self.read_arg_values() # Read options of GUI
        command = ["magnetism_preprocessing"]
        command.extend(arguments)
        
        self.thread = runMagnetism(command, path) # Run subprocess
        self.thread.out_signal.connect(self.update_out)
        self.thread.err_signal.connect(self.update_err)
        self.thread.finished.connect(self.program_finished)
        self.thread.start()
        
        
    @pyqtSlot(str)
    def update_out(self, output):   
        '''
        Update the output text.
        '''
        m_raise = re.search(r'\braise\b',output)
        m_error = re.search(r'\berror\b',output)
        if m_raise or m_error: 
            text_format = QTextCharFormat()
            text_format.setForeground(QColor(Qt.red))
            self.ui.notes.setCurrentCharFormat(text_format)
            self.ui.notes.append(output)
            text_format.setForeground(QColor(Qt.black))
            self.ui.notes.setCurrentCharFormat(text_format)
        else:
            text_format = QTextCharFormat()
            text_format.setForeground(QColor(Qt.black))
            self.ui.notes.setCurrentCharFormat(text_format)
            self.ui.notes.append(output)
    
        
    @pyqtSlot(str)
    def update_err(self, output):
        '''
        Update text with errors in color red
        '''
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(Qt.red))
        self.ui.notes.setCurrentCharFormat(text_format)
        self.ui.notes.append(output)


    @pyqtSlot()
    def program_finished(self):
        '''
        Enables the "GO!" button when the pipeline finishes
        '''
        self.ui.start_pushButton.setEnabled(True)  # Enable the start button
         
        
    def read_arg_values(self):
        '''
        Function to read argument values from the GUI.
        Change this function to add or modify new options
        
        Returns:
        --------
        arguments : list 
            List to store all the options and arguments of the GUI as it would be introduced in the magnetism pipeline.
        path : string
            Path of the data.  
        '''
        path = self.ui.path_lineEdit.text()
        if self.ui.selectTxt_radioButton.isChecked():
            file = self.ui.selectTxt_lineEdit.text()
            arguments = ['--txm', file]
        elif self.ui.selectDB_radioButton.isChecked():
            file = self.ui.selectDB_lineEdit.text()
            arguments = ['--db', file]
        elif self.ui.selectPattern_radioButton.isChecked():
            arguments = ['--pattern', self.ui.selectPattern_lineEdit.text()]
        else:
            arguments = []
        if self.ui.outlierThresh_checkBox.isChecked():
            arguments.extend(['--outlier_threshold',str(self.ui.outlierThresh_doubleSpinBox.value())])
        arguments.extend(['--delete_proc_files','True'] if self.ui.deleteProc_checkBox.isChecked() else ['--delete_proc_files','False'])
        arguments.extend(['--norm_ff','True'] if self.ui.normFF_checkBox.isChecked() else ['--norm_ff','False'])
        arguments.extend(['--sub_bg','True'] if self.ui.normBG_checkBox.isChecked() else ['--sub_bg','False'])
        arguments.extend(['--interpolate_ff','True'] if self.ui.interpFF_checkBox.isChecked() else ['--interpolate_ff','False'])
        arguments.extend(['--crop','True'] if self.ui.cropStack_checkBox.isChecked() else ['--crop','False'])
        arguments.extend(['--delete_prev_exec','True'] if self.ui.delPrevExec_checkBox.isChecked() else ['--delete_prev_exec','False'])
        if self.ui.saveAll_checkBox.isChecked(): arguments.append('--stack')

        return arguments,path
    
    def close_program(self):
        '''
        Close GUI
        '''
        self.close()
        return
    
    def select_file_dialog(self,wg,fileType):
        '''
        Open a dialog for selecting a file. Write the path in a widget.
        
        Parameters:
        -----------
        wg : lineEdit widget
            The filename selected is written in this widget. 
        fileType : string
            Type of the file to select.
            
        '''
        n = QFileDialog.getOpenFileName(None,"Select File",self.ui.path_lineEdit.text(),fileType)
        if n[0] != None:
            wg.setText("%s" % n[0])
            return n[0]
        else:
            return None
          
    def select_folder_dialog(self,wg):
        '''
        Open a dialog for selecting a folder. Write the path in a widget.
        
        Parameters:
        -----------
        wg : lineEdit widget
            The path selected is written in this widget. 
            
        '''
        n = QFileDialog.getExistingDirectory(None,"Select Path",options=QFileDialog.ShowDirsOnly)
        if n!= None:
            wg.setText("%s" % n)
            return n
        else:
            return None
        
    def visualization(self):
        '''
        Function to visualize a HDF5 stack using napari.
        It opens a dialog to select the HDF5. Then, it reads and displays the datasets inside it for the user to select one to visualize.
        '''
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.h5 *.hdf5)", options=options)

        if filename:
            dset_list = []
            with h5py.File(filename, "r") as f:
                # Read all dataset names and store it in keys
                keys = []
                f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
            self.visualization_select_dset(keys,filename)


    def visualization_select_dset(self,keys,filename):
        '''
        Function to display in a dialog the datasets inside the HDF5 file.
        
        Parameters:
        -----------
        keys : list
            Datasets to show the user to select one for visualization.
        filename : string
            Name of the HDF5 to visualize.
            
        '''
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Dataset")
        dialog.setGeometry(100, 100, 300, 200)
        
        l = QVBoxLayout()
        wg = QListWidget()
        wg.setSelectionMode(QAbstractItemView.MultiSelection) 
        wg.addItems(keys)
        l.addWidget(wg)
        
        open_napari_button = QPushButton("Open Napari")
        # open_napari_button.clicked.connect(lambda: self.open_napari_viewer(filename, wg.currentItem().text(),dialog))
        open_napari_button.clicked.connect(lambda: self.open_napari_viewer(filename, [item.text() for item in wg.selectedItems()],dialog))
        l.addWidget(open_napari_button)
        open_plt_button = QPushButton("Open Matplotlib")
        open_plt_button.clicked.connect(lambda: self.open_plt_viewer(filename, wg.currentItem().text(),dialog))
        l.addWidget(open_plt_button)

        dialog.setLayout(l)
        dialog.show()

    def open_napari_viewer(self, filename, dset,dialog):
        '''
        Function to open napari and show the dataset selected in a previous dialog.
        
        Parameters:
        -----------
        filename : string
            Name of the HDF5 to visualize.
        dset : string
            Name of the dataset selected for visualization.
        dialog : widget
            Previous dialog for selection of dataset.
            
        '''
        dialog.close()
        viewer = napari.Viewer()
        with h5py.File(filename, "r") as f:
            for i in dset:
                stack = f[i][...]
                stack[stack>1e20] = 0
                viewer.add_image(stack,contrast_limits=(stack.min(),stack.max()),name=i)
    
    def open_plt_viewer(self,filename,dset,dialog):
        '''
        Function to open matplotlib and show the dataset selected in a previous dialog.
        
        Parameters:
        -----------
        filename : string
            Name of the HDF5 to visualize.
        dset : string
            Name of the dataset selected for visualization.
        dialog : widget
            Previous dialog for selection of dataset.
            
        '''
        dialog.close()
        with h5py.File(filename, "r") as f:
            stack = f[dset][...]
        # self.v = visualization_plt(stack)
        self.thread = visualization_plt(stack) 
        self.thread.finished.connect(lambda: self.thread.deleteLater())  
        self.thread.start()

def main():
    app = QApplication(sys.argv)
    program = magnetism_GUI()
    program.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
