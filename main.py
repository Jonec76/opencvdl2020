from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import q1, q2
import sys

class GroupBox(QWidget):

	def __init__(self):
		QWidget.__init__(self)
		self.setWindowTitle("GroupBox")
		layout = QGridLayout()
		self.setLayout(layout)
		self.setFixedSize(1400, 800)
		
		total_widgets = []
		total_widgets.append(self.c1("1. Image Processing"))
		total_widgets.append(self.c2("2. Image Smoothing"))
		total_widgets.append(self.c3("3. Edge Detection"))
		total_widgets.append(self.c4("4. Transformation"))


		for idx, widget in enumerate(total_widgets):
			layout.addWidget(widget, 0, idx)

	def c1(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)

		b1 = QPushButton("1.1 Load Image")
		b2 = QPushButton("1.2 Color Seperation")
		b3 = QPushButton("1.3 Image Flipping")
		b4 = QPushButton("1.4 Blending")

		b1.clicked.connect(q1.load_image)
		b2.clicked.connect(q1.color_seperation)
		b3.clicked.connect(q1.image_flipping)
		b4.clicked.connect(q1.blending)
		
		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		vbox.addWidget(b4)
		return groupbox

	def c2(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)

		b1 = QPushButton("2.1 Median Filter")
		b2 = QPushButton("2.2 Gaussian Blur")
		b3 = QPushButton("2.3 Bilateral Filter")
		
		b1.clicked.connect(q2.median_filter)
		b2.clicked.connect(q2.gaussian_blur)
		b3.clicked.connect(q2.bilateral_filter)

		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		return groupbox

	def c3(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)

		b1 = QPushButton("3.1 Gaussian Blur")
		b2 = QPushButton("3.2 Sobel X")
		b3 = QPushButton("3.3 Sobel Y")
		b4 = QPushButton("3.4 Magnitude")

		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		vbox.addWidget(b4)
		return groupbox	

	def c4(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)
		r1 = self.get_c4_row("Rotation:", "deg")
		r2 = self.get_c4_row("Scaling: ", "")
		r3 = self.get_c4_row("Tx:      ", "pixel")
		r4 = self.get_c4_row("Ty:      ", "pixel")

		b1 = QPushButton("Transformation")

		vbox.addWidget(r1)
		vbox.addWidget(r2)
		vbox.addWidget(r3)
		vbox.addWidget(r4)
		vbox.addWidget(b1)
		return groupbox
		

	def get_c4_row(self, t1, t2):
		groupbox = QGroupBox()
		sub_layout = QGridLayout()
		groupbox.setLayout(sub_layout)

		L1 = QLabel(t1)
		E1 = QLineEdit()
		L2 = QLabel(t2)

		E1.setFixedWidth(160)

		sub_layout.addWidget(L1, 0, 0)
		sub_layout.addWidget(E1, 0, 1)
		sub_layout.addWidget(L2, 0, 2)

		return groupbox

app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())