
# Python   3.4
# PySide   1.2.4
# PyOpenGL 3.1.0

import sys
import numpy as np
from ctypes import sizeof, c_float, c_void_p

from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from OpenGL.GL.shaders import compileShader, compileProgram


class Window(QDialog):
    def __init__(self, parent=None):
        """
        This is a top-level window for displaying the OpenGL graphics.
        Use this as a container of the QGLWidget for better extensibility. We can add other widgets to interact with the graphics.
        """
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("OpenGL")
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
        self.resize(800, 600)

        self.glWidget = GLWidget(self)
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.glWidget, 0, 0, 1, 1)


class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        """
        QGLWidget provides functionality for displaying OpenGL graphics integrated into a Qt application.
        Reimplement 3 virtual functions in the subclass:
        1 paintGL() -- Renders the OpenGL scene.
        2 resizeGL() -- Sets up the OpenGL viewport, projection, ect. Gets called whenever the widget is resized.
        3 initializeGL() -- Sets up the OpenGL rendering context.
        """
        super().__init__(parent)

        self.parent = parent

        # set focus to catch keyPressEvent
        self.setFocus()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        pass

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.updateGL()

    def keyPressEvent(self, event):
        pass

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def wheelEvent(self, event):
        pass


def main():
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())


main()
