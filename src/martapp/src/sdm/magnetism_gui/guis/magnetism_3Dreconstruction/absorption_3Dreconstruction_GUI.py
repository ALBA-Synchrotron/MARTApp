import sys, os
import subprocess
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QListWidget, QVBoxLayout, QPushButton, QAbstractItemView
from PyQt5.QtCore import pyqtSlot,pyqtSignal, QThread, Qt
from PyQt5.QtGui import QTextCharFormat, QColor, QIcon, QPixmap
from .ui_absorption3Dreconstruction import Ui_magnetism_absorption3Dreconstruction
from datetime import datetime
import re, napari, h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,RangeSlider
import pkg_resources

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
        plt.show(block=False)
        plt.ioff()

    def update_clim(self,val):
        self.img.norm.vmin = val[0]
        self.img.norm.vmax = val[1]
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.ioff()
        
    def on_close(self,ev):
        plt.close('all')
        att = list(self.__dict__.keys())
        for i in att:
            delattr(self, i)
      
        
class absorption_3Dreconstruction_GUI(QDialog):
    '''
     Main pyqt5 GUI
    '''
    def __init__(self):
        '''
        Initialization and connections.
        '''
        super().__init__()
        self.ui = Ui_magnetism_absorption3Dreconstruction() # Open GUI
        self.ui.setupUi(self) # Set GUI
        
        # Include ALBA logos
        logo_path = pkg_resources.resource_filename("sdm.magnetism_gui", "resources/ALBA_positiu.ico")
        self.setWindowIcon(QIcon(logo_path))

        # Connections
        # Connections to read HDF5 file, its datasets and paths for tomo
        self.ui.selectFileHDF5Tomo_toolButton.clicked.connect(lambda: self.select_file_dialog(self.ui.fileHDF5Tomo_lineEdit,'HDF5 Files (*.hdf5)',wgPath=self.ui.outputPathTomo_lineEdit)) # Open dialog to select HDF5
        self.ui.fileHDF5Tomo_lineEdit.editingFinished.connect(lambda: self.add_dset_to_wg(self.ui.fileHDF5Tomo_lineEdit.text(),self.ui.dsetAbsorptionTomo_comboBox,key_word='AbsorptionTiltAligned'))
        self.ui.fileHDF5Tomo_lineEdit.editingFinished.connect(lambda: self.add_dset_to_wg(self.ui.fileHDF5Tomo_lineEdit.text(),self.ui.dsetAnglesTomo_comboBox,key_word='Angles'))
        self.ui.selectOutputPathTomo_toolButton.clicked.connect(lambda: self.select_folder_dialog(self.ui.outputPathTomo_lineEdit))
        self.ui.dsetAbsorptionTomo_comboBox.currentTextChanged.connect(lambda: self.read_dset(tomo=1))
        self.ui.recAxis_comboBox.currentIndexChanged.connect(lambda: self.read_dset(tomo=1))

        # Connections to read HDF5 file, its datasets and paths for tomo rot
        self.ui.selectFileHDF5TomoRot_toolButton.clicked.connect(lambda: self.select_file_dialog(self.ui.fileHDF5TomoRot_lineEdit,'HDF5 Files (*.hdf5)',wgPath=self.ui.outputPathTomoRot_lineEdit)) # Open dialog to select HDF5
        self.ui.fileHDF5TomoRot_lineEdit.editingFinished.connect(lambda: self.add_dset_to_wg(self.ui.fileHDF5TomoRot_lineEdit.text(),self.ui.dsetAbsorptionTomoRot_comboBox,key_word='AbsorptionTiltAligned'))
        self.ui.fileHDF5TomoRot_lineEdit.editingFinished.connect(lambda: self.add_dset_to_wg(self.ui.fileHDF5TomoRot_lineEdit.text(),self.ui.dsetAnglesTomoRot_comboBox,key_word='Angles'))
        self.ui.selectOutputPathTomoRot_toolButton.clicked.connect(lambda: self.select_folder_dialog(self.ui.outputPathTomoRot_lineEdit))
        self.ui.dsetAbsorptionTomoRot_comboBox.currentTextChanged.connect(lambda: self.read_dset(tomo=2))


        # Execute pipelines.
        self.ui.reconstructionTomo_pushButton.clicked.connect(lambda: self.start_program(tomo=1))
        self.ui.segmentationTomo_pushButton.clicked.connect(lambda: self.start_program(tomo=1,only_segmentation=True))

        self.ui.reconstructionTomoRot_pushButton.clicked.connect(lambda: self.start_program(tomo=2))
        self.ui.segmentationTomoRot_pushButton.clicked.connect(lambda: self.start_program(tomo=2,only_segmentation=True))

        # When clicking "Visualization" open output HDF5 file and select dataset for visualization with napari.
        self.ui.visualizationTomo_pushButton.clicked.connect(lambda: self.visualization(f"{self.ui.outputPathTomo_lineEdit.text()}/{self.ui.outputNameTomo_lineEdit.text()}_AbsorptionReconstruction.hdf5")) 
        self.ui.visualizationTomoRot_pushButton.clicked.connect(lambda: self.visualization(f"{self.ui.outputPathTomoRot_lineEdit.text()}/{self.ui.outputNameTomoRot_lineEdit.text()}_AbsorptionReconstruction.hdf5"))  # When clicking "Visualization" open dialog to select HDF5 for visualization.
        
        self.thread = None

    @pyqtSlot()
    def start_program(self,tomo=1,only_segmentation=False):
        '''
        Function to read the arguments and execute the magnetism pipeline.

        Parameters:
        -----------
        tomo : integer
            Choose between 1, to read arguments specifically for the first tomo or 2 for arguments for tomoRot.
        only_segmentation : Boolean
            Choose to add the flag to perform only segmentation in the arguments list.
        '''
        self.ui.notes.clear()  # Clear the text edit widget
        t = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.ui.notes.append(t+"\tRunning...")
        self.enable_button_group(self.ui.execution_buttonGroup,enabled=False)  # Disable the start button
        # Execute the magnetism program as a command-line program
        arguments,path = self.read_arg_values(tomo=tomo,only_segmentation=only_segmentation) # Read options of GUI
        command = ["magnetism_absorption3Dreconstruction"]
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
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(Qt.black))
        self.ui.notes.setCurrentCharFormat(text_format)


    @pyqtSlot()
    def program_finished(self):
        '''
        Enables the "GO!" button when the pipeline finishes
        '''
        self.enable_button_group(self.ui.execution_buttonGroup,enabled=True)  # Enable the start button

    def enable_button_group(self,group,enabled=True):
        '''
        Enables of disables a group of buttons of the GUI.

        Parameters:
        -----------
        group : string
            Object QButtonGroup to modify.
        enabled : Boolean
            Choose wether to enable or disable the buttons inside a group.
        '''
        for b in group.buttons():
            b.setEnabled(enabled)         
        
    def read_arg_values(self,tomo=1,only_segmentation=False):
        '''
        Function to read argument values from the GUI.
        Change this function to add or modify new options

        Parameters:
        -----------
        tomo : integer
            Choose between 1, to read arguments specifically for the first tomo or 2 for arguments for tomoRot.
        only_segmentation : Boolean
            Choose to add the flag to perform only segmentation in the arguments list.
        
        Returns:
        --------
        arguments : list 
            List to store all the options and arguments of the GUI as it would be introduced in the magnetism pipeline.
        path : string
            Path of the data.  
        '''
        if tomo == 1:
            path = self.ui.outputPathTomo_lineEdit.text()
            arguments = [self.ui.fileHDF5Tomo_lineEdit.text(),self.ui.dsetAbsorptionTomo_comboBox.currentText(),self.ui.dsetAnglesTomo_comboBox.currentText()]
            arguments.extend(['--output_filename',self.ui.outputNameTomo_lineEdit.text()])
            arguments.extend(['--segmentation_mode',self.ui.segmentationModeTomo_comboBox.currentText()])
            arguments.extend(['--threshold_val',self.ui.threshValTomo_comboBox.currentText()])
            if self.ui.negMaskTomo_checkBox.isChecked(): arguments.extend(['--mask_negative'])
        elif tomo == 2:
            path = self.ui.outputPathTomoRot_lineEdit.text()
            arguments = [self.ui.fileHDF5TomoRot_lineEdit.text(),self.ui.dsetAbsorptionTomoRot_comboBox.currentText(),self.ui.dsetAnglesTomoRot_comboBox.currentText()]
            arguments.extend(['--output_filename',self.ui.outputNameTomoRot_lineEdit.text()])
            arguments.extend(['--segmentation_mode',self.ui.segmentationModeTomoRot_comboBox.currentText()])
            arguments.extend(['--threshold_val',self.ui.threshValTomoRot_comboBox.currentText()])
            if self.ui.negMaskTomoRot_checkBox.isChecked(): arguments.extend(['--mask_negative'])
        # arguments.extend(['--mod_sxy',self.ui.modSxTomo_label.text()+' '+self.ui.modSyTomo_label.text()])
        arguments.extend(['--mod_sxy', str(self.ui.modSxTomo_spinBox.value()), str(self.ui.modSyTomo_spinBox.value())])
        arguments.extend(['--mod_sz',str(self.ui.modSzTomo_spinBox.value())])
        arguments.extend(['--pixel_size',str(self.ui.pixelSize_doubleSpinBox.value())])
        arguments.extend(['--n_iter',str(self.ui.nIter_spinBox.value())])
        if self.ui.LCFlag_checkBox.isChecked(): arguments.extend(['--lc_flag'])
        arguments.extend(['--reconstruction_axis',self.ui.recAxis_comboBox.currentText()])
        arguments.extend(['--simult_slcs',str(self.ui.simultSlcs_spinBox.value())])
        if self.ui.GPU_checkBox.isChecked(): arguments.extend(['--gpu'])
        if self.ui.saveProjMat_checkBox.isChecked(): arguments.extend(['--save_proj_matrices'])
        if self.ui.useProjMat_checkBox.isChecked(): arguments.extend(['--use_proj_matrices'])

        if only_segmentation: arguments.extend(['--only_segmentation'])
        return arguments,path
    
    def close_program(self):
        '''
        Close GUI
        '''
        self.close()
        return

    
    def select_file_dialog(self,wg,fileType,wgPath=None):
        '''
        Open a dialog for selecting a file. Write the path in a widget.
        
        Parameters:
        -----------
        wg : lineEdit widget
            The filename selected is written in this widget. 
        fileType : string
            Type of the file to select.
        wgFolderOut : lineEdit widget
            The path of the file selected is written in this widget.
        '''
        n = QFileDialog.getOpenFileName(None,"Select File","",fileType)
        if n[0] != None:
            wg.setText('%s' % n[0])
            wg.editingFinished.emit()     
            if wgPath:
                dir_name,_ = os.path.split(n[0]) 
                wgPath.setText(dir_name)
            return n[0]
        else:
            return None
          
    def select_folder_dialog(self,wg):
        '''
        Open a dialog for selecting a folder. Write the path in a widget.
        
        Parameters:
        -----------
        wg : lineEdit widget.
            The path selected is written in this widget. 
            
        '''
        n = QFileDialog.getExistingDirectory(None,"Select Path","",options=QFileDialog.ShowDirsOnly)
        if n!= None:
            wg.setText("%s" % n)
            wg.editingFinished.emit()   
            return n
        else:
            return None
        
    def add_dset_to_wg(self,file,widget,key_word=''):
        '''
         Read the datasets in the file and make them options in the comboBox fixed.
        '''
        widget.clear()
        if os.path.exists(file):
            with h5py.File(file,'r') as f:
                # Read all dataset names and store it in keys
                keys = []
                f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
                # Add keys to combobox
                widget.addItems(keys)
            if key_word in keys:
                widget.setCurrentIndex(keys.index(key_word))
            else:
                widget.setCurrentIndex(0)
            # self.read_dset_fixed()
        else:
            self.ui.notes.clear()  # Clear the text edit widget
            t = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            self.ui.notes.append(t+"\tHDF5 File not found")

        return

    def read_dset(self,tomo=1): 
        '''
        Read dataset of first tilt series and obtain parameters for reconstruction:
         size of reconstruction space and default simultaneous slices
        '''
        if tomo == 1:
            file = self.ui.fileHDF5Tomo_lineEdit.text()
            dset = self.ui.dsetAbsorptionTomo_comboBox.currentText()
        else:
            file = self.ui.fileHDF5TomoRot_lineEdit.text()
            dset = self.ui.dsetAbsorptionTomoRot_comboBox.currentText()
        try:
            with h5py.File(file,'r') as f:
                data = f[dset][...]
        except:
            return
        if len(data.shape) < 3:
            return
        self.ui.modSxTomo_spinBox.setValue((data.shape[1]))
        self.ui.modSyTomo_spinBox.setValue((data.shape[2]))
        found = False
        if self.ui.recAxis_comboBox.currentText()=="YTilt":
            if data.shape[1] > 20:
                for i in range(10,1,-1):
                    if data.shape[1]%i == 0:
                        self.ui.simultSlcs_spinBox.setValue(i)
                        found = True
                        return
        elif self.ui.recAxis_comboBox.currentText()=="XTilt":
            if data.shape[2] > 20:
                for i in range(10,1,-1):
                    if data.shape[2]%i == 0:
                        self.ui.simultSlcs_spinBox.setValue(i)
                        found = True
                        return
        if not found:
            self.ui.simultSlcs_spinBox.setValue(1)
        return
            

        
        
    def visualization(self,filename=None):
        '''
        Function to visualize a HDF5 stack using napari.
        It opens a dialog to select the HDF5. Then, it reads and displays the datasets inside it for the user to select one to visualize.
        '''
        if not filename or not os.path.exists:
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
    program = absorption_3Dreconstruction_GUI()
    program.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
