from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import q5
import sys

class test_dialogue(QWidget):
	def __init__(self):
		super().__init__()
		layout = QVBoxLayout()
		self.label = QLabel("Test Image Index")
		self.E1 = QLineEdit()
		self.btn = QPushButton("Inference")

		layout.addWidget(self.label)
		layout.addWidget(self.E1)
		layout.addWidget(self.btn)

		self.btn.clicked.connect(lambda: q5.test_image(self.E1.text()))
		self.setLayout(layout)

class GroupBox(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		self.setWindowTitle("GroupBox")
		layout = QGridLayout()
		self.setLayout(layout)
		self.setFixedSize(300, 300)
		
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
		b4 = QPushButton("4. Show Accuracy")
		b5 = QPushButton("5. Test")


		b1.clicked.connect(q5.show_train_image)
		b2.clicked.connect(q5.show_params)
		b3.clicked.connect(q5.show_model_structure)
		# b4.clicked.connect(q5.show_accuracy)
		b5.clicked.connect(self.on_pushButton_clicked)

		vbox.addWidget(b1)
		vbox.addWidget(b2)
		vbox.addWidget(b3)
		# vbox.addWidget(b4)
		vbox.addWidget(b5)
		return groupbox
	def on_pushButton_clicked(self):
		self.w = test_dialogue()
		self.w.show()
	

app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())