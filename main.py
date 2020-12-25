from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import q1
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

		for idx, widget in enumerate(total_widgets):
			layout.addWidget(widget, 0, idx)

	def c1(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)

		b1 = QPushButton("1.1 Load image")
		b2 = QPushButton("1.2 Count coins")
		x = 5
		l1 = QLabel(self)
		# l1.resize(200,100)
		l1.setText("There are _ conis in coin01.jpg\nThere are _ conis in coin02.jpg")
		b1.clicked.connect(q1.find_contour)
		b2.clicked.connect(lambda y: q1.count_coins(l1))
		
		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(l1)

		return groupbox

	

app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())