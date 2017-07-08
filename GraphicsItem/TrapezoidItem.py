import sys

from PySide.QtCore import *
from PySide.QtGui import *

import numpy as np


class Dialog(QDialog):
    def __init__(self):
        super(Dialog, self).__init__()

        self.Parameters = ["Width", "Height", "f", "a", "b", "alpha"]
        self.defaultValues = [100, 50, 25, 25, 50, 70]
        self.colorOptions = ["Red", "Black", "Blue", "Green"]

        self.setupUI()
        self.setupGraphics()
        self.setupConnection()

    def setupUI(self):
        self.setWindowTitle("Trapezoid")
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(1000, 600)
        self.setSizeGripEnabled(True)

        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        # dialog layout
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # splitter between left QFrame and right QFrame
        self.splitter = QSplitter(self)
        self.splitter.setOrientation(Qt.Horizontal)

        # Left QFrame
        # Three rows: top QGraphicsView, middle spacerItem, bottome QLabel
        self.frame1 = QFrame(self.splitter)

        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(3)  # left frame width : right frame width = 3 : 1
        self.frame1.setSizePolicy(sizePolicy1)

        self.layout1 = QGridLayout(self.frame1)
        self.layout1.setContentsMargins(0, 0, 0, 0)

        # Top QGraphicsView
        self.View = QGraphicsView(self.frame1)
        # Middle spacerItem
        spacerItem = QSpacerItem(20, 5, QSizePolicy.Minimum, QSizePolicy.Maximum)
        # Bottom (left QCheckbox + right QLabel)
        self.labelCoordinate = QLabel("X: , Y:")
        self.labelCoordinate.setAlignment(Qt.AlignCenter)

        self.layout1.addWidget(self.View, 0, 0, 1, 1)
        self.layout1.addItem(spacerItem, 1, 0, 1, 1)
        self.layout1.addWidget(self.labelCoordinate, 2, 0, 1, 1)

        # Right QFrame
        self.frame2 = QFrame(self.splitter)
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(1)
        self.frame2.setSizePolicy(sizePolicy2)

        self.layout2 = QGridLayout(self.frame2)
        self.layout2.setContentsMargins(0, 0, 0, 0)

        # 5 pairs of QLabel and QLineEdit for 5 parameters and the input widget
        self.lineEditList = []
        for i, parameter in enumerate(self.Parameters):
            label = QLabel(parameter)
            label.setAlignment(Qt.AlignCenter)

            lineEdit = QLineEdit()
            defaultValue = self.defaultValues[i]
            lineEdit.setText("{}".format(defaultValue))
            self.lineEditList.append(lineEdit)

            self.layout2.addWidget(label, i, 0, 1, 1)
            self.layout2.addWidget(lineEdit, i, 1, 1, 1)

        # 2 pairs of QLabel and QCombobox for 2 colors
        labelInner = QLabel("Inner Color")
        comboInner = QComboBox()
        comboInner.addItems(self.colorOptions)

        labelOuter = QLabel("Outer Color")
        comboOuter = QComboBox()
        comboOuter.addItems(self.colorOptions)
        self.comboboxList = [comboInner, comboOuter]

        self.layout2.addWidget(labelInner, i + 1, 0, 1, 1)
        self.layout2.addWidget(comboInner, i + 1, 1, 1, 1)
        self.layout2.addWidget(labelOuter, i + 2, 0, 1, 1)
        self.layout2.addWidget(comboOuter, i + 2, 1, 1, 1)

        self.layout.addWidget(self.splitter, 0, 0, 1, 1)

    def setupGraphics(self):
        self.Scene = QGraphicsScene()
        self.Scene.setSceneRect(0, 0, 100, 100)
        self.View.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Coordinate origin at top-left
        self.View.setScene(self.Scene)

        self.trapezoidItem = TrapezoidalItem(self.View, self)
        self.Scene.addItem(self.trapezoidItem)

    def setupConnection(self):
        # When changing value in the lineEdits, or colors from the combobox, the plot will update
        for lineEdit in self.lineEditList:
            lineEdit.editingFinished.connect(self.trapezoidItem.UpdateData)

        for combo in self.comboboxList:
            combo.currentIndexChanged.connect(self.trapezoidItem.UpdateData)

    def showEvent(self, event):
        self.trapezoidItem.resizeEvent(event)

    def resizeEvent(self, event):
        self.trapezoidItem.resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            # Move focus away from lineEdits and the 'editingFinished' will emit
            self.View.setFocus()


class TrapezoidalItem(QGraphicsItem):
    def __init__(self, parent=None, topParent=None):
        super().__init__()

        self.parent = parent
        self.topParent = topParent  # QDialog

        # Enable mouse tracking
        self.setAcceptHoverEvents(True)

        # default values
        self._width = 100
        self._height = 50
        self._f = 25
        self._a = 25
        self._b = 50
        self._alpha = 70

        self._fRatio = self._f / self._width
        self._aRatio = self._a / self._width
        self._bRatio = self._b / self._width

        self._innerColor = Qt.red
        self._outerColor = Qt.red

        self.colorMap = {"Red": Qt.red,
                         "Black": Qt.black,
                         "Blue": Qt.blue,
                         "Green": Qt.green}

        self.UpdatePlot()

    def UpdatePlot(self):
        # Calculate the margin beyond the trapezoid
        leftBoundary = self.parent.width() * 0.05
        botBoundary = self.parent.height() - 80  # bottom margin is 80 pixels

        # left margin 5%, right margin 60 pixels
        HSpace = self.parent.width() - leftBoundary - 60
        VSpace = HSpace * self._height / self._width  # HSpace / VSpace = width / height

        # Coordinate origin lies on the top left of the GraphicsView
        # 4 points marking the drawing area, a rectangle.
        self.PointTop1 = QPointF(leftBoundary,          botBoundary - VSpace)  # top left
        self.PointTop2 = QPointF(leftBoundary + HSpace, botBoundary - VSpace)  # top right
        self.PointBot1 = QPointF(leftBoundary,          botBoundary)  # bottom left
        self.PointBot2 = QPointF(leftBoundary + HSpace, botBoundary)  # bottom right

        # 4 points marking the trapezoid
        b1 = leftBoundary + HSpace * self._fRatio
        b2 = b1 + HSpace * self._bRatio
        self.PointB1 = QPointF(b1, botBoundary)  # lower base left point
        self.PointB2 = QPointF(b2, botBoundary)  # lower base right point

        # alpha is in degree, turn it into radian first
        a1 = VSpace / np.tan(np.deg2rad(self._alpha)) + b1
        a2 = a1 + HSpace * self._aRatio
        self.PointA1 = QPointF(a1, botBoundary - VSpace)  # upper base left point
        self.PointA2 = QPointF(a2, botBoundary - VSpace)  # upper base right point

        # an arc for Alpha text, calculate the Rect for painting
        radius = max(VSpace / 10, 20)
        self.RectAlpha = QRectF(b1 - radius, botBoundary - radius,
                                radius * 2, radius * 2)  # b1 is the center

        # position for 'alpha' text
        ax = b1 + 1.2 * radius * np.cos(np.deg2rad(self._alpha / 2))
        ay = botBoundary - 1.2 * radius * np.sin(np.deg2rad(self._alpha / 2))
        self.RectAlphaTxt = QRectF(ax, ay - 25, 40, 50)

        # position for f/a/b text
        self.PointTxtF = QPointF(leftBoundary + HSpace * self._fRatio / 2 - 5, botBoundary + 20)
        self.PointTxtA = QPointF((a1 + a2) / 2 - 5, botBoundary - VSpace - 10)
        self.PointTxtB = QPointF((b1 + b2) / 2 - 5, botBoundary + 20)

        # |----- h -----| vertical notation on the right
        # 1    2   3    4
        self.PointTxtH = QPointF(leftBoundary + HSpace + 18, botBoundary - VSpace / 2)

        self.PointH1 = self.PointTop2 + QPointF(20,  2)
        self.PointH2 = self.PointTxtH + QPointF(2, -28)  # 2 make the line vertical
        self.PointH3 = self.PointTxtH + QPointF(2, 15)  # -28, 15 adjust the space away from text
        self.PointH4 = self.PointBot2 + QPointF(20, -2)

        # |----- w -----| horizontal notation at the bottom
        # 1    2   3    4
        self.PointTxtW = QPointF(leftBoundary + HSpace / 2, botBoundary + 45)

        self.PointW1 = self.PointBot1 + QPointF(2, 40)
        self.PointW2 = self.PointTxtW + QPointF(-20, -5)  # -20, 30 horizontal space away from text
        self.PointW3 = self.PointTxtW + QPointF(30, -5)  # -5 make the line horizontal
        self.PointW4 = self.PointBot2 + QPointF(-2, 40)

        self.update(self.boundingRect())

    def UpdateData(self):
        try:
            w, h, f, a, b, alpha = self.topParent.lineEditList
            inner, outer = self.topParent.comboboxList

            self._width = float(w.text())
            self._height = float(h.text())
            self._fRatio = float(f.text()) / self._width
            self._aRatio = float(a.text()) / self._width
            self._bRatio = float(b.text()) / self._width
            self._alpha = float(alpha.text())

            self._innerColor = self.colorMap[inner.currentText()]
            self._outerColor = self.colorMap[outer.currentText()]
        except:
            error = QMessageBox.information(self.topParent, "Error", "Invalid input")
            return

        self.UpdatePlot()

    def boundingRect(self):
        # Make boundingRect the whole view, not just the trapezoid.
        # Because there are labels beyond that region.
        cell = QRectF(0, 0, self.parent.width(), self.parent.height())
        return cell

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen()
        pen.setWidth(2)
        pen.setStyle(Qt.SolidLine)
        pen.setColor(Qt.black)
        painter.setPen(pen)

        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)

        # Ridge media
        painter.setBrush(self._innerColor)
        painter.drawConvexPolygon([self.PointA1, self.PointA2, self.PointB2, self.PointB1])

        # Groove media
        painter.setBrush(self._outerColor)

        painter.drawConvexPolygon([self.PointTop1, self.PointA1, self.PointB1, self.PointBot1])
        painter.drawConvexPolygon([self.PointA2, self.PointTop2, self.PointBot2, self.PointB2])

        # an arc for angle text
        painter.drawArc(self.RectAlpha, 0, self._alpha * 16)

        # alpha text. DrawText does not support HTML, so use QTextDocument
        alphaTxt = QTextDocument()
        alphaTxt.setHtml("<html><head/><body><p style='font-size:20px'>&alpha;</p></body></html>")

        painter.save()
        painter.translate(self.RectAlphaTxt.left(), self.RectAlphaTxt.top())
        alphaTxt.drawContents(painter, QRectF(
            0, 0, self.RectAlphaTxt.width(), self.RectAlphaTxt.height()))
        painter.restore()

        # draw f/a/b text outside the rectangle
        painter.drawText(self.PointTxtF, "f")
        painter.drawText(self.PointTxtA, "a")
        painter.drawText(self.PointTxtB, "b")

        # The following makes the notation on the right:                                              ---
        # text                                        |
        painter.drawText(self.PointTxtH, "h")
        # top |                                       |
        painter.drawLine(self.PointH1, self.PointH2)
        # bot |                                       h
        painter.drawLine(self.PointH3, self.PointH4)
        #             |
        painter.drawLine(self.PointH1 - QPointF(5, 0), self.PointH1 +
                         QPointF(5, 0))  # top---      |
        painter.drawLine(self.PointH4 - QPointF(5, 0), self.PointH4 +
                         QPointF(5, 0))  # bot---     ---

        # The following makes the notation on the top: |--------- w ---------|
        painter.drawText(self.PointTxtW, "w")
        painter.drawLine(self.PointW1, self.PointW2)  # left  ---------
        painter.drawLine(self.PointW3, self.PointW4)  # right ---------

        painter.drawLine(self.PointW1 - QPointF(0, 5), self.PointW1 + QPointF(0, 5))  # left  |
        painter.drawLine(self.PointW4 - QPointF(0, 5), self.PointW4 + QPointF(0, 5))  # right |

    def resizeEvent(self, event):
        self.UpdatePlot()

    def hoverMoveEvent(self, event):
        # Rectangle
        left = self.PointBot1.x()
        bot = self.PointBot1.y()
        right = self.PointTop2.x()
        top = self.PointTop2.y()

        # Cursor position based on the GraphicsScene coordinate
        x, y = event.pos().x(), event.pos().y()

        # Coordinate, based on the Trapezoid coordinate, bottom-left point as coordinate origin
        coordX = self._width * (x - left) / (right - left)
        coordY = self._height * (bot - y) / (bot - top)

        self.topParent.labelCoordinate.setText("X: {:.4g}, Y: {:.4g}".format(coordX, coordY))


def main():
    app = QApplication(sys.argv)

    window = Dialog()
    window.show()

    app.exec_()


if __name__ == '__main__':
    main()
