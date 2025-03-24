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

from .ui_ManualImageAlignment import Ui_ManualImageAlignment
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys, h5py
import numpy as np
import SimpleITK as sitk

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.image as mpimg
from matplotlib.widgets import RangeSlider
import pkg_resources
import matplotlib.pyplot as plt


class MplCanvas(Canvas):
    '''
    Matplotlib canvas class to create figure
    '''
    
    def __init__(self,parent=None):
        super().__init__() 
        self.fig = Figure()
        self.fig.tight_layout()
        Canvas.__init__(self, self.fig)
        self.ax_pair = self.fig.add_subplot(121)
        self.ax_pair.imshow(np.zeros((100,100)))
        self.ax_diff = self.fig.add_subplot(1,2,2)
        self.ax_diff.sharex(self.ax_pair)
        self.ax_diff.sharey(self.ax_pair)
        self.ax_diff.imshow(np.zeros((100,100)))
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        self.ax_pair.axis('off')
        self.ax_diff.axis('off')
        self.ax_pair_clim = self.fig.add_axes([0.05, 0.25, 0.0225, 0.63])
        self.clim_pair = RangeSlider(ax=self.ax_pair_clim,
                            label='Clim',
                            valmin=0.0,
                            valmax=1.0,
                            orientation='vertical'
                            )
        self.ax_diff_clim = self.fig.add_axes([0.95, 0.25, 0.0225, 0.63])
        self.clim_diff = RangeSlider(ax=self.ax_diff_clim,
                            label='Clim',
                            valmin=0.0,
                            valmax=1.0,
                            orientation='vertical'
                            )

class ManualImageAlignment(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ManualImageAlignment()
        self.ui.setupUi(self)
        
        # Include ALBA logos
        logo_path = pkg_resources.resource_filename("sdm.magnetism_gui", "resources/ALBA_positiu.ico")
        self.setWindowIcon(QIcon(logo_path))

        # Add Matplotlib canvas widget to show images.
        self.canvas = MplCanvas(self)
        self.ui.imshow_horizontalLayout.addWidget(self.canvas)
        self.toolbar_canvas = NavigationToolbar(self.canvas,self)
        self.ui.figures_verticalLayout.addWidget(self.toolbar_canvas)

        # Create slider to change color range.
        self.canvas.clim_pair.on_changed(self.update_sliders)
        self.canvas.clim_diff.on_changed(self.update_sliders)
                
        # Initialization variables
        self.stack_fixed = None
        self.stack_moving = None

        # Connections
        self.ui.SelectFileFixed.clicked.connect(self.open_hdf5_fixed)
        self.ui.dsetFixed.currentIndexChanged.connect(self.read_dset_fixed)
        self.ui.nimg_fixed.valueChanged.connect(self.update_images)
        self.ui.SelectFileMoving.clicked.connect(self.open_hdf5_moving)
        self.ui.dsetMoving.currentIndexChanged.connect(self.read_dset_moving)
        self.ui.nimg_moving.valueChanged.connect(self.change_idx_moving)
        self.ui.Export.clicked.connect(self.export_data)

        # Conmections to update images
        # self.ui.nimg_moving.valueChanged.connect(self.update_images)
        self.ui.shiftX.valueChanged.connect(self.update_images)
        self.ui.shiftY.valueChanged.connect(self.update_images)
        self.ui.scaleX.valueChanged.connect(self.update_images)
        self.ui.scaleY.valueChanged.connect(self.update_images)
        self.ui.rotation.valueChanged.connect(self.update_images)
        self.ui.intensityFactor.valueChanged.connect(self.update_images)

        # Connections to change steps.
        self.ui.shiftXStep.textChanged.connect(lambda:self.change_step(self.ui.shiftX,self.ui.shiftXStep))
        self.ui.shiftYStep.textChanged.connect(lambda:self.change_step(self.ui.shiftY,self.ui.shiftYStep))
        self.ui.scaleXStep.textChanged.connect(lambda:self.change_step(self.ui.scaleX,self.ui.scaleXStep))
        self.ui.scaleYStep.textChanged.connect(lambda:self.change_step(self.ui.scaleY,self.ui.scaleYStep))
        self.ui.rotationStep.editingFinished.connect(lambda:self.change_step(self.ui.rotation,self.ui.rotationStep))
        self.ui.intensityFactorStep.textChanged.connect(lambda:self.change_step(self.ui.intensityFactor,self.ui.intensityFactorStep))
        
        # Connections to arrows to generate displacements.
        self.keyPressEvent = self.activate_shifts_arrows

    def open_hdf5_fixed(self):
        '''
        Open dialog to select moving HDF5. Write the name in the label.
        '''
        file = QtWidgets.QFileDialog.getOpenFileName(None,"Select File","","*.hdf5")
        if file[0] != None:
            self.ui.fileFixed.setText("%s" % file[0])
            self.add_dset_to_wg_fixed(file[0])
        return file

            
    def add_dset_to_wg_fixed(self,file):
        '''
         Read the datasets in the file and make them options in the comboBox fixed.
        '''
        if file:
            self.ui.dsetFixed.clear()
            with h5py.File(file,'r') as f:
                # Read all dataset names and store it in keys
                keys = []
                f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
                # Add keys to combobox
                self.ui.dsetFixed.addItems(keys)
            self.ui.dsetFixed.setCurrentIndex(0)
            self.read_dset_fixed() 
        return 
        
    def open_hdf5_moving(self):
        '''
        Open dialog to select moving HDF5. Write the name in the label.
        '''
        file = QtWidgets.QFileDialog.getOpenFileName(None,"Select File","","*.hdf5")
        if file[0] != None:
            self.ui.fileMoving.setText("%s" % file[0])
            self.add_dset_to_wg_moving(file[0])
        return file

            
    def add_dset_to_wg_moving(self,file):
        '''
         Read the datasets in the file and make them options in the comboBox moving.
        '''
        self.ui.dsetMoving.clear()
        with h5py.File(file,'r') as f:
            # Read all dataset names and store it in keys
            keys = []
            f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
            # Add keys to combobox
            self.ui.dsetMoving.addItems(keys)
        self.ui.dsetMoving.setCurrentIndex(0)
        self.read_dset_moving() 
        return 
        

    def read_dset_fixed(self):
        '''
        Read dataset of fixed HDF5. If it is a stack, change the maximum value of 
        the spiner to the size of the stack.
        '''
        
        file = self.ui.fileFixed.text()
        dset = self.ui.dsetFixed.currentText()
        with h5py.File(file,'r') as f:
            stack_fixed = f[dset][()] # Read fixed stack
        if len(np.shape(stack_fixed)) < 3: # Dataset contains only one image
            self.stack_fixed = np.zeros((1,np.shape(stack_fixed)[0],np.shape(stack_fixed)[1]))
            self.stack_fixed[0,:,:] = stack_fixed
        else:
            self.stack_fixed = stack_fixed
        self.ui.nimg_fixed.setMaximum(np.shape(self.stack_fixed)[0]-1) # Fix spinner to maximum value
            
        if self.stack_fixed is not None:
            self.update_images()

        
    def read_dset_moving(self):
        '''
        Read dataset of moving HDF5. If it is a stack, change the maximum value of 
        the spiner to the size of the stack. Create a dictionary 'data' to store the transformations for all images in stack.
        '''
        file = self.ui.fileMoving.text()
        dset = self.ui.dsetMoving.currentText()
        with h5py.File(file,'r') as f:
            stack_moving = f[dset][()] # Read moving stack
        if len(np.shape(stack_moving)) < 3: # Dataset contains only one image
            self.stack_moving = np.zeros((1,np.shape(stack_moving)[0],np.shape(stack_moving)[1]))
            self.stack_moving[0,:,:] = stack_moving
        else:
            self.stack_moving = stack_moving
        self.ui.nimg_moving.setMaximum(np.shape(self.stack_moving)[0]-1) # Fix spinner to maximum value

        # Initialization: create dictionary to store all the transformation values for each image in the stack
        data = np.array([0.0,0.0,1.0,1.0,0.0,1.0])# Parameters: shiftx, shifty, scalex, scaley, rotation, intensity
        self.data = np.tile(data,(np.shape(self.stack_moving)[0],1)) # Dictionary for all images in the stack
        
        self.change_idx_moving()
        
        

    def change_idx_moving(self):
        '''
        Function to update the recover the transformations ofthe new image moving when changing value in spiner.
        It writes the value of the widgets from the dictionary 'data'.
        '''
        list_wg = [self.ui.shiftX, self.ui.shiftY, self.ui.scaleX, self.ui.scaleY, self.ui.rotation, self.ui.intensityFactor]
        idx = self.ui.nimg_moving.value()

        for i in range(0,len(list_wg)):
            # Block signals
            list_wg[i].blockSignals(True)
            # Write value
            list_wg[i].setValue(self.data[idx][i])
            # Unblock signals
            list_wg[i].blockSignals(False)
        if self.stack_fixed is not None:
            self.canvas.ax_pair.clear()
            self.canvas.ax_diff.clear()
            self.update_images()


    def update_images(self):
        '''
        When touching any parameter, update images.
        Perform transformations to img_moving and show both images.
        '''
        try:
            img_fixed = self.stack_fixed[self.ui.nimg_fixed.value(),:,:] # Read fixed image
            idx_moving = self.ui.nimg_moving.value() # Read idx for moving image
            img_moving = self.stack_moving[idx_moving,:,:] # Read moving image
        except:
            return
        # Calculate centre of transformation for fixed and moving images with the convention of SimpleITK
        R_fixed = np.multiply([(img_fixed.shape[1] - 1) / 2, (img_fixed.shape[0] - 1) / 2], -1)
        R_moving = np.multiply([(img_moving.shape[1] - 1) / 2, (img_moving.shape[0] - 1) / 2], -1)

        # Update imshowpair sliders
        img_pair = img_fixed+img_moving
        self.canvas.clim_pair.valmax = img_pair.max()
        self.canvas.clim_pair.valmin = img_pair.min()
        self.canvas.clim_pair.ax.set_ylim(img_pair.min(),img_pair.max())
 

        # Update imshowdiff slider
        img_diff = np.log(img_fixed)-np.log(img_moving)
        img_diff[np.isnan(img_diff) | np.isinf(img_diff)] = 0
        self.canvas.clim_diff.valmax = img_diff.max()
        self.canvas.clim_diff.valmin = img_diff.min()
        self.canvas.clim_diff.ax.set_ylim(img_diff.min(),img_diff.max())

        self.update_data() # Read values of widgets for current image moving
        tform = self.obtain_tform(idx_moving) # Obtain transformation.
        img_tform = self.image_warp(img_moving,R_moving,img_fixed,R_fixed,tform[:-1])*tform[-1] # Apply transformation.

        self.imshowimgs(img_fixed,img_tform)


    def imshowimgs(self,img_fixed,img_moving):
        '''
        Function to show both images overlapping and the difference between logarithms.
        '''
        # Define images for plotting
        img_pair = img_fixed+img_moving
        img_diff = np.log(img_fixed)-np.log(img_moving)
        img_diff[np.isnan(img_diff) | np.isinf(img_diff)] = 0

        # Check if an image is already drawn on the canvas.
        img_canvas = False
        for artist in self.canvas.ax_diff.get_children():
            if isinstance(artist, mpimg.AxesImage):
                img_canvas = True
                break
        # To keep the zoom, save limits of axis before clearing.
        if img_canvas:
            xlim = self.canvas.ax_pair.get_xlim()
            ylim = self.canvas.ax_pair.get_ylim()
        else: # If no image is drawn, the limits are the shape of images
            xlim = (0.0, img_fixed.shape[1])
            ylim = (0.0, img_fixed.shape[0])

        self.canvas.ax_pair.clear()
        self.canvas.ax_pair.imshow(img_pair,cmap='gray',alpha=0.5,
                                         vmin=self.canvas.clim_pair.val[0],
                                         vmax=self.canvas.clim_pair.val[1])
        self.canvas.ax_pair.set_xlim(xlim)
        self.canvas.ax_pair.set_ylim(ylim)
        self.canvas.draw()

        self.canvas.ax_diff.clear()
        self.canvas.ax_diff.imshow(img_diff,cmap='gray',
                                         vmin=self.canvas.clim_diff.val[0],
                                         vmax=self.canvas.clim_diff.val[1])
        # self.canvas.ax_diff.set_xlim(xlim)
        # self.canvas.ax_diff.set_ylim(ylim)
        self.canvas.draw()

    def update_sliders(self,val):
        self.update_images()

    def obtain_tform(self,idx):
        '''
        Obtain geometrical transformation matrix from dictionary 'data' based on the idx selected.
        The irst 4 elements are the rotation matrix, the next 2 the translation and the last one is the intensity factor
        '''
        cosd = lambda a: np.cos(np.radians(a))
        sind = lambda a: np.sin(np.radians(a))
        
        tform = np.array([1.0,0.0,0.0,1.0,0.0,0.0,1.0])
        tform[0] = cosd(self.data[idx][4])*self.data[idx][2]
        tform[1] = -sind(self.data[idx][4])
        tform[2] = sind(self.data[idx][4])
        tform[3] = cosd(self.data[idx][4])*self.data[idx][3]
        tform[4] = -self.data[idx][0]
        tform[5] = -self.data[idx][1]
        tform[6] = self.data[idx][5]
        return tform.tolist()
    

    def update_data(self):
        '''
        Read the value of the widgets and update the dictionary 'data'.
        '''
       
        idx = self.ui.nimg_moving.value()
        self.data[idx][0] = self.ui.shiftX.value()
        self.data[idx][1] = self.ui.shiftY.value()
        self.data[idx][2] = self.ui.scaleX.value()
        self.data[idx][3] = self.ui.scaleY.value()
        self.data[idx][4] = self.ui.rotation.value()
        self.data[idx][5] = self.ui.intensityFactor.value()
        return
    

    def image_warp(self,img_moving, R_moving, img_fixed, R_fixed, tform):
        """
        Transform the input image img_moving according to the geometric
        transformation tform. The center of transformation is the center of
        the image. It is used SimpleITK.

        Parameters:
        -----------
        img_moving : ndarray, shape (x, y)
            2D array image to which the transformation is applied.
        R_moving : ndarray, shape (2,)
            1D array positions of the origin of the moving image using
            the convention of SimpleITK.
        img_fixed : ndarray, shape (x, y)
            2D array reference image (used by SimpleITK, although the reason is
            not clear).
        R_fixed : ndarray, shape (2,)
            1D array positions of the origin of the fixed image using
            the convention of SimpleITK.
        tform : list, shape (6,)
            1x6 list representing the geometric transformation to apply.
            The first four numbers are the rotation matrix.
            The last two numbers are the translation.

        Returns:
        --------
        img_tform : ndarray, shape (x, y)
            2D array image after performing the transformation.
            Same size as img_moving.
        """

        # Load images into sitk
        img_fixed_itk = sitk.GetImageFromArray(img_fixed)
        img_moving_itk = sitk.GetImageFromArray(img_moving)
        img_fixed_itk.SetOrigin(R_fixed)
        img_moving_itk.SetOrigin(R_moving)

        # Generate transformation
        tform_itk = sitk.AffineTransform(2)
        tform_itk.SetCenter((0, 0))
        tform_itk.SetMatrix(tform[:4])
        tform_itk.SetTranslation(tform[4:])

        # Apply transformation
        img_tform_itk = sitk.Resample(
            img_moving_itk,
            img_fixed_itk,
            tform_itk,
            sitk.sitkLinear,
            0.0,
            img_fixed_itk.GetPixelID(),
        )
        # Retrieve images into numpy array
        img_tform = sitk.GetArrayFromImage(img_tform_itk)
        return img_tform
    

    def change_step(self,wg,wg_step):
        '''
        Change step of the widget wg to the value written in wg_step
        '''
        if wg_step.text()!='':
            wg.setSingleStep(float(wg_step.text()))


    def export_data(self):
        '''
        Transform all moving stack according to the dictionary 'data'. 
        Options to store result in same hdf5 with the same dataset name and suffix _ManualAli or create a new hdf5.
        It also can export the Absorption2DAlign and MagneticSignal2DAlign
        '''
        stack_out = np.zeros_like(self.stack_moving) # Initialization
        tform = np.zeros((np.shape(self.stack_moving)[0],7),dtype=np.float64)
        for idx in range(0,len(self.stack_moving)):
            # Calculate centre of transformation for fixed and moving images with the convention of SimpleITK to save
            R_fixed = np.multiply([(self.stack_fixed.shape[2] - 1) / 2, (self.stack_fixed.shape[1] - 1) / 2], -1)
            R_moving = np.multiply([(self.stack_moving.shape[2] - 1) / 2, (self.stack_moving.shape[1] - 1) / 2], -1)
            tform[idx] = self.obtain_tform(idx) # Obtain tform
            stack_out[idx,:,:] = self.image_warp(self.stack_moving[idx,:,:],R_moving,self.stack_fixed[idx,:,:],R_fixed,tform[idx][:-1])*tform[idx][-1] # Perform transformation.
                
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog  # Use native file dialog if preferred
        if self.ui.exportNewHDF5_radioButton.isChecked(): # To create a new HDF5
            dset_fixed = self.ui.dsetFixed.currentText()
            dset_moving = self.ui.dsetMoving.currentText() # Read dataset of hdf5
            dset_tform = f"{dset_moving}_tform"
            
            file, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Create New HDF5 File", "", "HDF5 Files (*.hdf5);;All Files (*)", options=options)
            file = f'{file}.hdf5'
            with h5py.File(file,'a') as f: # Create file
                if dset_tform not in f:
                    f.create_dataset(dset_tform,data=tform,dtype=np.float64)
                else:
                    f[dset_tform][...] = tform
        else:
            # Overwrite XMCD file.
            # Find xmcd file in folder.
            dset_fixed = '2DAlignedPositiveStack'
            dset_moving = '2DAlignedNegativeStack'
            file, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Select XMCD HDF5 File to overwrite", "", "HDF5 Files (*.hdf5);;All Files (*)", options=options)
            with h5py.File(file,'a') as f: # Write dataset
                stack_fixed = self.stack_fixed
                stack_fixed[stack_fixed<=0] = 1
                stack_out[stack_out<=0] = 1
                absorption = -np.add(np.log(stack_fixed),np.log(stack_out))/2
                xmcd = -np.add(np.log(stack_fixed),-np.log(stack_out))/2
                with h5py.File(file,'a') as f: # Write dataset
                    if "Absorption2DAligned" in f:
                        del f["Absorption2DAligned"] 
                    f.create_dataset("Absorption2DAligned",data=absorption,dtype=np.float64)

                    if "MagneticSignal2DAligned" in f:
                        del f["MagneticSignal2DAligned"]
                    f.create_dataset("MagneticSignal2DAligned",data=xmcd,dtype=np.float64)                       
               
        with h5py.File(file,'a') as f: # Write dset moving and fixed
                if dset_fixed in f:
                    del f[dset_fixed]
                f.create_dataset(dset_fixed,data=self.stack_fixed,dtype=np.float64)
                if dset_moving in f:
                    del f[dset_moving]
                f.create_dataset(dset_moving,data=stack_out,dtype=np.float64)
        
        # Reset everything back to beginning
        self.stack_moving = stack_out        
        self.ui.shiftX.setValue(0.0)
        self.ui.shiftY.setValue(0.0)
        self.ui.scaleX.setValue(1.0)
        self.ui.scaleY.setValue(1.0)
        self.ui.rotation.setValue(0.0)
        self.ui.intensityFactor.setValue(1.0)        
        # Initialization: create dictionary to store all the transformation values for each image in the stack
        data = np.array([0.0,0.0,1.0,1.0,0.0,1.0])# Parameters: shiftx, shifty, scalex, scaley, rotation, intensity
        self.data = np.tile(data,(np.shape(self.stack_moving)[0],1)) # Dictionary for all images in the stack

    def activate_shifts_arrows(self,event):
        '''
        Associate a key to a button. Specifically, the arrows will perform shifts in X and Y.
        '''
        if event.key() == Qt.Key_Right: # Shift in +X when pressing right arrow.
            val = self.ui.shiftX.value()
            self.ui.shiftX.setValue(val+float(self.ui.shiftXStep.text()))
        elif event.key() == Qt.Key_Left: # Shift in -X when pressing left arrow.
            val = self.ui.shiftX.value()
            self.ui.shiftX.setValue(val-float(self.ui.shiftXStep.text()))
        elif event.key() == Qt.Key_Up: # Shift in +Y when pressing up arrow.
            val = self.ui.shiftY.value()
            self.ui.shiftY.setValue(val+float(self.ui.shiftYStep.text()))
        elif event.key() == Qt.Key_Down: # Shift in -Y when pressing down arrow.
            val = self.ui.shiftY.value()
            self.ui.shiftY.setValue(val-float(self.ui.shiftYStep.text()))
        self.update_images()



def main():
    app = QtWidgets.QApplication(sys.argv)
    program = ManualImageAlignment()
    program.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()