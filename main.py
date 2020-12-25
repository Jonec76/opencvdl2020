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
		total_widgets.append(self.c2("2. Calibration"))

		for idx, widget in enumerate(total_widgets):
			layout.addWidget(widget, 0, idx)

	def c1(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)

		b1 = QPushButton("1.1 Load image")
		b2 = QPushButton("1.2 Count coins")
		l1 = QLabel(self)
		l1.setText("There are _ conis in coin01.jpg\nThere are _ conis in coin02.jpg")
		
		b1.clicked.connect(q1.find_contour)
		b2.clicked.connect(lambda y: q1.count_coins(l1))
		
		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(l1)

		return groupbox

	def c2(self, title):
		groupbox = QGroupBox(title)
		sub_groupbox = QGroupBox("2.3 Find Extrinsic")
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)
		sub_vbox = QVBoxLayout()
		sub_groupbox.setLayout(sub_vbox)
		
		b1 = QPushButton("2.1 Find Corners")
		b2 = QPushButton("2.2 find Intrinsic")
		b4 = QPushButton("2.4 Find Distortion")

		# For sub groupbox
		label = QLabel("Select image")
		b3 = QPushButton("2.3 Find Extrinsic")
		listWidget = QComboBox()
		
		for i in range(1, 16):
			listWidget.addItem(str(i))
		
		b1.clicked.connect(q2.find_corners)
		b2.clicked.connect(q2.find_intrinsic)
		b3.clicked.connect(lambda y: q2.find_extrinsic(listWidget.currentText()))
		b4.clicked.connect(q2.find_distortion)

		sub_vbox.addWidget(label)
		sub_vbox.addWidget(listWidget)
		sub_vbox.addWidget(b3)
		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(sub_groupbox)
		vbox.addWidget(b4)


		return groupbox

	

app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())