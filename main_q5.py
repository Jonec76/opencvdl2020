from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import q5
import sys

class GroupBox(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		self.setWindowTitle("GroupBox")
		layout = QGridLayout()
		self.setLayout(layout)
		self.setFixedSize(400, 800)
		
		total_widgets = []
		total_widgets.append(self.c1("Q5"))

		for idx, widget in enumerate(total_widgets):
			layout.addWidget(widget, 0, idx)

	def c1(self, title):
		groupbox = QGroupBox(title)
		vbox = QVBoxLayout()
		groupbox.setLayout(vbox)

		b1 = QPushButton("1. Show Train Image")
		b2 = QPushButton("2. Show Hyperparameters")
		b3 = QPushButton("3. Show Model Structure")


		b1.clicked.connect(q5.show_train_image)
		# b2.clicked.connect(q5.show_params)
		b3.clicked.connect(q5.show_model_structure)
		
		
		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		return groupbox
	

app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())