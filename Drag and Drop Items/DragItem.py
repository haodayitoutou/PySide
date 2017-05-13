import sys

from PySide.QtCore import *
from PySide.QtGui import *


class Radish(QGraphicsTextItem):
    """
    This is a rectangular item displaying some text. You can drag this to a Pit.
    """
    def __init__(self, txt, xRatio, yRatio, parent):
        super(Radish, self).__init__()

        self.setPlainText(txt)

        self.xRatio = xRatio #resize with the GraphicsView
        self.yRatio = yRatio

        self.parent = parent
        self.setCursor(Qt.OpenHandCursor)
        self.UpdateSize()

    def UpdateSize(self):  #update size
        self.Width = self.parent.width() * self.xRatio
        self.Height = self.parent.height() * self.yRatio

    def boundingRect(self):
        rect = QRectF( -self.Width * 0.5, -self.Height * 0.5, self.Width, self.Height)
        return rect

    def shape(self):
        #set the shape so the whole rectangle will accept the mousePressEvent
        painterPath = QPainterPath()
        painterPath.addRect( self.boundingRect() )
        return painterPath

    def paint(self, painter, option, widget):
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)

        pen = QPen()
        pen.setWidth(1)
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)

        painter.drawRect( self.boundingRect() ) #draw the boundary
        painter.drawText( self.boundingRect(), Qt.AlignCenter, self.toPlainText() )  #draw the text

    def resizeEvent(self, event):
        self.UpdateSize()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        pos1 = QPointF( event.screenPos() )
        pos2 = QPointF( event.buttonDownScreenPos(Qt.LeftButton) )

        if QLineF(pos1, pos2).length() < QApplication.startDragDistance():
            #mouse movement is too little, ignore this event
            return

        #transfer the data
        mime = QMimeData()
        mime.setText( self.toPlainText() )

        pixmap = QPixmap(self.Width, self.Height)
        pixmap.fill(Qt.yellow)

        painter = QPainter(pixmap)
        painter.translate( self.Width*0.5, self.Height*0.5)
        painter.setRenderHint(QPainter.Antialiasing)
        self.paint(painter, None, None)
        painter.end()
        pixmap.setMask(pixmap.createHeuristicMask() )

        drag = QDrag( event.widget() )
        drag.setMimeData(mime)
        drag.setPixmap(pixmap)
        drag.setHotSpot( QPoint(self.Width*0.5, self.Height*0.5) )
        drag.exec_()
        self.setCursor(Qt.OpenHandCursor)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)


class Pit(QGraphicsObject, QGraphicsItem):
    """
    This item can accept a dropEvent and update its text.
    """
    def __init__(self, xRatio, yRatio, parent):
        super(Pit, self).__init__()

        self.xRatio = xRatio
        self.yRatio = yRatio

        self.parent = parent
        self.setAcceptDrops(True)

        self.Txt = ""
        self.status = 0
        self.UpdateSize()

    def UpdateSize(self):  #update size
        self.Width = self.parent.width() * self.xRatio
        self.Height = self.parent.height() * self.yRatio

    def boundingRect(self):
        rect = QRectF( -self.Width * 0.5, -self.Height * 0.5, self.Width, self.Height)
        return rect

    def paint(self, painter, option, widget):
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)

        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)

        #change background color based on its status

        if self.status == 1:   #accepted an text, but has not been validated
            brush.setColor(Qt.yellow)
        elif self.status == 2: #validation passed
            brush.setColor(Qt.green)
        elif self.status == 3: #validation failed. Text does not match with the image
            brush.setColor(Qt.red)
        else:                 #default status
            brush.setColor(Qt.white)

        painter.setBrush(brush)

        painter.drawRect( self.boundingRect() )
        painter.drawText( self.boundingRect(), Qt.AlignCenter, self.Txt )

    def resizeEvent(self, event):
        self.UpdateSize()

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.setAccepted(True)
            self.update()
        else:
            event.setAccepted(False)

    def dragLeaveEvent(self, event):
        self.update()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            self.Txt = event.mimeData().text()
            self.status = 1
        self.update()


class IconItem(QGraphicsPixmapItem):

    def __init__(self, file, parent):
        super(IconItem, self).__init__()

        self.pixmap = QPixmap(file)
        self.parent = parent

    def boundingRect(self):
        rect = QRect( -self.width * 0.5, -self.height * 0.5, self.width, self.height)
        return rect

    def paint(self, painter, options, widget):
        painter.drawPixmap( self.boundingRect(), self.pixmap )

    def resizeEvent(self, event):
        self.width = self.pixmap.scaledToHeight( self.parent.height()*0.1 ).width()
        self.height = self.pixmap.scaledToHeight( self.parent.height()*0.1 ).height()


class View(QGraphicsView):

    def __init__(self, parent=None):
        super(View, self).__init__(parent)

        self.parent = parent

        self.Scene = QGraphicsScene(self)
        self.Scene.setSceneRect(0, 0, 100, 100)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setScene(self.Scene)

        #text
        self.words = [
            "As High As Honor",
            "Fire and Blood",
            "Growing Strong",
            "Hear me roar",
            "Ours is the Fury",
            "We Do Not Sow",
            "Winter is coming"
            ]

        #images
        self.icons = [
            "Arryn.png",
            "Baratheon.png",
            "Greyjoy.png",
            "Lannister.png",
            "Stark.png",
            "Targaryen.png",
            "Tyrell.png"
            ]

        self.wordItems = []
        self.iconItems = []
        self.pitItems  = []

        for i in range(7):
            wordItem = Radish(self.words[i], xRatio=0.25, yRatio=0.1, parent=self)
            self.wordItems.append(wordItem)
            self.Scene.addItem(wordItem)

            iconItem = IconItem(self.icons[i], self)
            self.iconItems.append(iconItem)
            self.Scene.addItem(iconItem)

            pitItem = Pit(xRatio=0.25, yRatio=0.1, parent=self)
            self.pitItems.append(pitItem)
            self.Scene.addItem(pitItem)


    def resizeEvent(self, event):
        #decide the items' positions
        xPos1 = self.width() * 0.25
        xPos2 = self.width() * 0.55
        xPos3 = self.width() * 0.75

        for i in range( len(self.wordItems) ):
            yPos = self.height() * 0.125 * (i+1)

            wordItem = self.wordItems[i]
            wordItem.resizeEvent(event)
            wordItem.setPos(xPos1, yPos)

            iconItem = self.iconItems[i]
            iconItem.resizeEvent(event)
            iconItem.setPos(xPos2, yPos)

            pitItem = self.pitItems[i]
            pitItem.resizeEvent(event)
            pitItem.setPos(xPos3, yPos)


class Dialog(QDialog):
    def __init__(self):
        super(Dialog, self).__init__()

        self.setWindowTitle("One Pit for one Radish")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(600, 800)
        self.setMinimumWidth(550)

        self.layout = QGridLayout(self)

        self.view = View(self)
        self.layout.addWidget(self.view, 0, 0, 1, 4)

        self.check = QPushButton("Check")
        self.check.clicked.connect(self.Validate)
        self.layout.addWidget(self.check, 1, 3, 1, 1)

        self.rightAnswer = [0, 4, 5, 3, 6, 1, 2]

    def Validate(self):
        for i in range(7):
            pitItem = self.view.pitItems[i]

            if pitItem.Txt:
                answer = self.view.words.index( pitItem.Txt )

                if answer == self.rightAnswer[i]:
                    status = 2  #right answer, show green
                else:
                    status = 3  #wrong, show red
            else:
                status = 3  #no answer, show red

            pitItem.status = status
            pitItem.update()



def main():
    app = QApplication(sys.argv)

    window = Dialog()
    window.show()

    app.exec_()


if __name__ == '__main__':
    main()
