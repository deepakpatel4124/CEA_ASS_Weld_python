from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog 
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
import sys
import os
import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import plotly.express as px
from plotly.offline import plot
from scipy.signal import hilbert

class start(QMainWindow):
    
    
    def __init__(self):
        super(start, self).__init__()
        
        loadUi(r"C:\Users\Deepak\Desktop\Python FMC-TFM\FMCTFM\start.ui", self)
        
    
        self.linear_array.clicked.connect(self.tfm_opener)
        self.with_wedge.clicked.connect(self.wedge_opener)
        
    def tfm_opener(self):
        self.line_arr = tfm()
        self.line_arr.show()
    
    def wedge_opener(self):
        self.wedge = tfm_wedge()
        self.wedge.show()
    
    
    
class tfm_wedge(QMainWindow):
    
    hil_image = None
    fw = 1
    
    def __init__(self):
        super(tfm_wedge, self).__init__()
        
        loadUi(r"C:\Users\Deepak\Desktop\Python FMC-TFM\FMCTFM\la_wedge.ui", self)
        
        #Gui Handler
        self.flat_wedge_inp.setChecked(True)
        self.file_open.clicked.connect(self.show_file_dialog)
        self.run.clicked.connect(self.runn)
        self.plotly_graph.clicked.connect(self.plotly_graph1)
        self.save_image.clicked.connect(self.saveImage)
        self.flat_wedge_inp.clicked.connect(self.flat_wedge1)
        self.angle_wedge_inp.clicked.connect(self.ang_wedge1)
   
        
    def runn(self):
        self.run.setText('Running')
        self.main()
        
        
    def flat_wedge1(self):
        self.angle_wedge_inp.setChecked(False)
        self.wedge_angle_inp.setEnabled(False)
        self.fst_ele_x.setEnabled(False)
        self.fst_ele_y.setEnabled(False)
        self.wedge_height.setEnabled(True)
        tfm_wedge.fw = 1
    
    def ang_wedge1(self):
        self.flat_wedge_inp.setChecked(False)
        self.wedge_angle_inp.setEnabled(True)
        self.fst_ele_x.setEnabled(True)
        self.fst_ele_y.setEnabled(True)
        self.wedge_height.setEnabled(False)
        tfm_wedge.fw = 0
        
    def show_file_dialog(self):
        # create a file dialog object
        file_dialog = QFileDialog()

        # set the file dialog options
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)

        # get the file path from the user
        if file_dialog.exec_():
            file_path_ = file_dialog.selectedFiles()[0]
            self.file_path_input.setText(file_path_)
            self.sub_main()
            
    def sub_main(self):
        tfm.file_path = self.file_path_input.text()
        file= np.fromfile(tfm.file_path, dtype="<i2", sep='')
        tfm.transducer_num = int(file[55])
        tfm.data_points = int(file[63])
        self.data_points_input.setText(str(tfm.data_points))
        self.tran_num.setText(str(tfm.transducer_num))
        
    def saveImage(self):
        # get the file path of the image in the current directory
        tempFilePath = os.path.join(os.getcwd(), "temp.png")
   
        # get the folder path to save the image file
        folderPath = QFileDialog.getExistingDirectory(self, 'Select Directory')
   
        if folderPath:
            # create a new file name with extension
            newFileName, _ = QFileDialog.getSaveFileName(self, 'Save Image', folderPath, "Image Files (*.png *.jpg *.bmp *.gif)")
   
            if newFileName:
                # move the image file from the current directory to the new folder with the new file name
                os.rename(tempFilePath, newFileName) 
    
    
    
    def get_matrix(self,file):
        transducer_num = int(file[55])
        data_points=file[64-1]
        file=file[64:]
        matrix  = np.array([])
        for i in range(transducer_num):
            start=(data_points+10)*i*transducer_num+i*10
            end=start+(data_points+10)*transducer_num
            asc = file[start:end]
            d=np.array([])
            d = np.append(d, asc)
            d = d.reshape(transducer_num,data_points+10)
            matrix = np.append(matrix, d[:,:data_points])
        
        matrix = matrix.reshape(transducer_num,transducer_num,data_points)
        
        return matrix
    
    def transducer_position(self,pitch,wedge_angle,transducer_num):
        trans = {}
        xx = float(self.fst_ele_x.text())
        yy = float(self.fst_ele_y.text())
        
        trans[0] = (xx/1000,yy/1000)
    
        for i in range(1,transducer_num):
    
            trans[i] = (trans[i-1][0]+pitch*math.cos(wedge_angle*math.pi/180),trans[i-1][1]-pitch*math.sin(wedge_angle*math.pi/180))
    
        return trans

    def new_wedge_tfm_image(self, pitch, transducer_num, data_points, matrix, roi, pix_size, V1, V2, frequency, wedge_angle):
        
        
        start_time = time.time()
        
        if tfm_wedge.fw == 1:
            hui = float(self.wedge_height.text())/1000
            transducer_position = [((transducer - transducer_num/2+0.5)*pitch , hui*(-1)) for transducer in range(transducer_num)]             
            transducers = dict(enumerate(transducer_position)) 
            
        else:
            
            transducers = self.transducer_position(pitch,wedge_angle,transducer_num)
        
        width = int(round((roi[1] - roi[0]) / pix_size))
        height = int(round((roi[3] - roi[2]) / pix_size))
        
        image_array = np.zeros((width, height))
        
        sep = 0.2/1000
        
        for y in range(height):
            
#            progress = int((y+1)/height*100)
#            
#            self.progressBar.setValue(progress)
            
            Y=(y*pix_size+roi[2])
            
            for x in range(width):
                
                X=(x*pix_size+roi[0])
        
                   
                for t, transmitter in transducers.items():
                    
                    if round(X,4) == round(transmitter[0],4):
                        xt = X
                        tof1 = (((transmitter[0]-xt)**2+transmitter[1]**2)**0.5)/V1 + (((X-xt)**2+(Y)**2)**0.5)/V2
                        prop_time1 = tof1
                    else:
                        xt = np.arange(min(X,transmitter[0]),max(X,transmitter[0]),sep)
                        tof1 = (((transmitter[0]-xt)**2+transmitter[1]**2)**0.5)/V1 + (((X-xt)**2+(Y)**2)**0.5)/V2
                        prop_time1 = tof1.min()
    
                    for r, receiver in transducers.items():    
                        
                        if round(X,4) == round(receiver[0],4):
                            xd = X
                            tof2 = (((receiver[0]-xd)**2+receiver[1]**2)**0.5)/V1 + (((X-xd)**2+(Y)**2)**0.5)/V2
                            prop_time2 = tof2
                        else:
                            xd = np.arange(min(X,receiver[0]),max(X,receiver[0]),sep)
                            tof2 = (((receiver[0]-xd)**2+receiver[1]**2)**0.5)/V1 + (((X-xd)**2+(Y)**2)**0.5)/V2
                            prop_time2 = tof2.min()
                        
                        prop_time = prop_time1 + prop_time2
    
                        adata = int(prop_time*frequency)
                        image_array[x][y] += matrix[t][r][adata]
                        
            elapsed_time = time.time()-start_time
            time_remain = elapsed_time*(height-y+1)/(y+1)
            self.timer.setText("Estimated time remaining: {:.2f} seconds".format(time_remain))
    
        return image_array.transpose()
        
    
    def hilbert_transform(self, signals):

        envelopes = np.zeros_like(signals)
    
        # Loop through each transmitter-receiver pair and compute the envelope signal
        for i in range(signals.shape[1]):
    
            x = signals[:,i]
            x_hilbert = hilbert(x)

            # Compute the absolute value of the complex signal to obtain the envelope
            x_env = np.abs(x_hilbert)
            
            envelopes[:,i] = x_env

        return envelopes
    
    

    def plot1(self, image, roi):
    
        # Define the color map
        color_list = [(0.0, 'white'),   # white
                      (0.33, 'blue'),   # blue
                      (0.67, 'yellow'), # yellow
                      (1.0, 'red')]     # red
        cmap = colors.LinearSegmentedColormap.from_list('mycmap', color_list)
    
        # Create the plot using the custom color map
       
        plt.imshow(image, cmap=cmap,  extent=[roi[0]*1000, roi[1]*1000, roi[3]*1000, roi[2]*1000])
        plt.colorbar()

        
        plt.savefig('temp.png', bbox_inches='tight')  
        plt.clf()  # clear the figure
        pixmap = QtGui.QPixmap('temp.png')
        pixmap2 = pixmap.scaled(self.plt_disp.width(), self.plt_disp.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        item = QtWidgets.QGraphicsPixmapItem(pixmap2)
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(item)
        self.plt_disp.setScene(scene)
        
    def plotly_graph1(self):
        
        colors = [(0.0, 'rgb(255, 255, 255)'),  # white
        (0.33, 'rgb(0, 0, 255)'),    # blue
        (0.67, 'rgb(255, 255, 0)'),  # yellow
        (1.0, 'rgb(255, 0, 0)')]  #white

        fig = px.imshow(tfm.hil_image, color_continuous_scale=colors)

        plot(fig, auto_open=True)


    def main(self):
    
        ## Inputs
        
        
        file_path = self.file_path_input.text()
        pix_size = float(self.pix_size_input.text())/1000
        pitch = float(self.pitch_input.text())/1000
        V2 = float(self.velocity_input.text())
        frequency = float(self.freq.text())*1e6
        V1 = float(self.wedge_velocity_inp.text())
        wedge_angle = float(self.wedge_angle_inp.text())
        
        file= np.fromfile(file_path, dtype="<i2", sep='') 
        data_points = file[64-1]
        transducer_num = int(file[55])
          
        
        #Roi
        min_X = float(self.min_x.text())
        max_X = float(self.max_x.text())
        min_Y = float(self.min_y.text())
        max_Y = float(self.max_y.text())
        
        
        roi = np.array([min_X,max_X,min_Y,max_Y])/1000
        
        
        #TFM Calculations
        matrix = self.get_matrix(file) 
        image = self.new_wedge_tfm_image(pitch, transducer_num, data_points, matrix, roi, pix_size, V1, V2, frequency, wedge_angle)
        tfm.hil_image = self.hilbert_transform(image)
        self.plot1(tfm.hil_image, roi)
        self.run.setText('Run')














    
    
   ##############################################LA TFM############################## 

class tfm(QMainWindow):
    
    hil_image = None
    
    def __init__(self):
        super(tfm, self).__init__()
        
        loadUi(r"C:\Users\Deepak\Desktop\Python FMC-TFM\FMCTFM\fmctfm.ui", self)
        
        
        #Gui Handler
        self.file_open.clicked.connect(self.show_file_dialog)
        self.run.clicked.connect(self.runn)
        self.plotly_graph.clicked.connect(self.plotly_graph1)
        self.save_image.clicked.connect(self.saveImage)
        
    def runn(self):
        self.run.setText('Running')
        self.main()
        
        
    def show_file_dialog(self):
        # create a file dialog object
        file_dialog = QFileDialog()

        # set the file dialog options
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)

        # get the file path from the user
        if file_dialog.exec_():
            file_path_ = file_dialog.selectedFiles()[0]
            self.file_path_input.setText(file_path_)
            self.sub_main()
            
    def saveImage(self):
        # get the file path of the image in the current directory
        tempFilePath = os.path.join(os.getcwd(), "temp.png")

        # get the folder path to save the image file
        folderPath = QFileDialog.getExistingDirectory(self, 'Select Directory')

        if folderPath:
            # create a new file name with extension
            newFileName, _ = QFileDialog.getSaveFileName(self, 'Save Image', folderPath, "Image Files (*.png *.jpg *.bmp *.gif)")

            if newFileName:
                # move the image file from the current directory to the new folder with the new file name
                os.rename(tempFilePath, newFileName) 
    
    def sub_main(self):
        tfm.file_path = self.file_path_input.text()
        file= np.fromfile(tfm.file_path, dtype="<i2", sep='')
        tfm.transducer_num = int(file[55])
        tfm.data_points = int(file[63])
        self.data_points_input.setText(str(tfm.data_points))
        self.tran_num.setText(str(tfm.transducer_num))
            
    def get_matrix(self,file):
        transducer_num = int(file[55])
        data_points=file[64-1]
        file=file[64:]
        matrix  = np.array([])
        for i in range(transducer_num):
            start=(data_points+10)*i*transducer_num+i*10
            end=start+(data_points+10)*transducer_num
            asc = file[start:end]
            d=np.array([])
            d = np.append(d, asc)
            d = d.reshape(transducer_num,data_points+10)
            matrix = np.append(matrix, d[:,:data_points])
        
        matrix = matrix.reshape(transducer_num,transducer_num,data_points)
        
        return matrix
            
    def data_index(self, transducers, pixel_ind, velocity, frequency):

        i = 0
        dist = {}
        for i, tr in transducers.items():
            dist[i] = ((np.sqrt((tr[0]-pixel_ind[0])**2+(tr[1]-pixel_ind[1])**2))/ velocity)*frequency
            i+=1
    
        return dist
    
    def plotly_graph1(self):
        
        colors = [(0.0, 'rgb(255, 255, 255)'),  # white
        (0.33, 'rgb(0, 0, 255)'),    # blue
        (0.67, 'rgb(255, 255, 0)'),  # yellow
        (1.0, 'rgb(255, 0, 0)')]  #white

        fig = px.imshow(tfm.hil_image, color_continuous_scale=colors)

        plot(fig, auto_open=True)
        
    def plot1(self, image, roi):
    
        # Define the color map
        color_list = [(0.0, 'white'),   # white
                      (0.33, 'blue'),   # blue
                      (0.67, 'yellow'), # yellow
                      (1.0, 'red')]     # red
        cmap = colors.LinearSegmentedColormap.from_list('mycmap', color_list)
    
        # Create the plot using the custom color map
        global im
        im = plt.imshow(image, cmap=cmap,  extent=[roi[0]*1000, roi[1]*1000, roi[3]*1000, roi[2]*1000])
        plt.colorbar()

        
        plt.savefig('temp.png', bbox_inches='tight')  
        plt.clf()  # clear the figure
        pixmap = QtGui.QPixmap('temp.png')
        pixmap2 = pixmap.scaled(self.plt_disp.width(), self.plt_disp.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        item = QtWidgets.QGraphicsPixmapItem(pixmap2)
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(item)
        self.plt_disp.setScene(scene)
        
        
                
    
    def fast_tfm(self,start_time, pitch, data_points , transducer_num, matrix, roi, pix_size, velocity, frequency):

        transducer_position = [((transducer - transducer_num/2+0.5)*pitch , 0) for transducer in range(transducer_num)] 
        
        transducers = dict(enumerate(transducer_position)) 
        
        width = int(round((roi[1] - roi[0]) / pix_size))
        height = int(round((roi[3] - roi[2]) / pix_size))
    
        
        ## creating x cordinate matrixnew_wedge_tfm_image(pitch, transducer_num, matrix, min_X, max_X, min_Y, max_Y, pix_size, V1, frequency,sep=0.2/1000):
        x_pixel = np.linspace(roi[0],roi[1], width)
        
    
       
        ## creating y cordinate matrixvji hncdgjiurn5ti
        y_pixel = np.linspace(roi[2],roi[3],height)
        
        
        # Cretating ROI Cordinates
        pixel_ind = np.meshgrid(x_pixel,y_pixel)
    
    
        ## getting data
        image_array = np.zeros((height, width))
        
        # matrix = hilbert_transform(matrix,frequency,)
        
        data = matrix
    
        ## calculating data points
        dp = self.data_index(transducers, pixel_ind, velocity, frequency)
    
    
        for t in range(transducer_num):
            
#            progress = int((t+1)/transducer_num*100)
#            
#            self.progressBar.setValue(progress)
            
            for r in range(transducer_num):
    
                ai = dp[t]+dp[r]
                
                bi = ai.reshape(width*height).astype(int)
    
                amp = np.take(data[t][r], bi)
    
                amp = amp.reshape(height,width)
    
                image_array += amp
                
            bhil = self.hilbert_transform(image_array)
            self.plot1(bhil, roi)
            
            elapsed_time = time.time()-start_time
            time_remain = elapsed_time*(transducer_num-t+1)/(t+1)
            self.timer.setText("Estimated time remaining: {:.2f} seconds".format(time_remain))
        
        self.timer.setText("")
        return image_array
    
    def hilbert_transform(self, signals):

        envelopes = np.zeros_like(signals)
    
        # Loop through each transmitter-receiver pair and compute the envelope signal
        for i in range(signals.shape[1]):
    
            x = signals[:,i]
            x_hilbert = hilbert(x)

            # Compute the absolute value of the complex signal to obtain the envelope
            x_env = np.abs(x_hilbert)
            
            envelopes[:,i] = x_env

        return envelopes

    def main(self):
    
        ## Inputs
        
        start_time = time.time()
        file_path = self.file_path_input.text()
        pix_size = float(self.pix_size_input.text())/1000
        pitch = float(self.pitch_input.text())/1000
        velocity = float(self.velocity_input.text())
        frequency = float(self.freq.text())*1e6
        
        
        file= np.fromfile(file_path, dtype="<i2", sep='') 
        data_points = file[64-1]
        transducer_num = int(file[55])
          
        
        #Roi
        min_X = float(self.min_x.text())
        max_X = float(self.max_x.text())
        min_Y = float(self.min_y.text())
        max_Y = float(self.max_y.text())
        
        
        roi = np.array([min_X,max_X,min_Y,max_Y])/1000
        
        
        #TFM Calculations
        matrix = self.get_matrix(file) 
        image = self.fast_tfm(start_time, pitch, data_points , transducer_num, matrix, roi, pix_size, velocity, frequency)
        tfm.hil_image = self.hilbert_transform(image)
        self.plot1(tfm.hil_image, roi)
        self.run.setText('Run')


        
        
if __name__=="__main__":
    
    app = QApplication(sys.argv)
    ui = start()
    ui.show()
    app.exec_()
        
            