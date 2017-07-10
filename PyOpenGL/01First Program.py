
#Python   3.4
#PySide   1.2.4
#PyOpenGL 3.1.0

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
        Use this as a container of the QGLWidget for better extensibility.
        We can add other widgets to interact with the graphics in the future.
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

        #set focus to catch keyPressEvent
        self.setFocus()

        #Define parameters
        self.NumVAOs = 1
        self.NumBuffers = 1
        self.vPosition = 0
        self.NumVertices = 6
        self.float_size = sizeof(c_float)
        
    def initializeGL(self):
        #Initialize vertex-array objects
        self.VAOs = glGenVertexArrays(self.NumVAOs) #return a name for use as VAO
        glBindVertexArray(self.VAOs)                #create VAO and assign the name

        self.vertices = np.array([
            -0.90, -0.90,
             0.85, -0.90,
            -0.90,  0.85,
             0.90, -0.85,
             0.90,  0.90,
            -0.85,  0.90], dtype='float32')

        #Allocate vertex-buffer objects
        self.VBOs = glGenBuffers(self.NumBuffers) #return a name for use as VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs)  #create VBO and assign the name
        
        #Load data into a buffer object
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW) #transfer vertex data from objects into buffer

        #Initialize vertex and fragment shaders
        strVertexShader = """
        #version 430 core                        //version of OpenGL Shading Language to use
        layout(location = 0) in vec4 vPosition;  //shader variable

        void main() {
            gl_Position = vPosition;            //special vertex-shader output
        }
        """
        strFragmentShader = """
        #version 430 core

        out vec4 fColor;

        void main() {
            fColor = vec4(0.0, 0.0, 1.0, 1.0);  //RGB color space, ranging from [0, 1]
        }
        """

        self.program = compileProgram(
            compileShader(strVertexShader, GL_VERTEX_SHADER),
            compileShader(strFragmentShader, GL_FRAGMENT_SHADER)
            )

        #Associate variables in a vertex shader with data stored in VBO
        stride = 2 * self.float_size
        offset = c_void_p(0 * self.float_size)
        glVertexAttribPointer(self.vPosition, 2, GL_FLOAT, GL_FALSE, stride, offset)
        glEnableVertexAttribArray(self.vPosition)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        glBindVertexArray(self.VAOs) #select the vertex array as vertex data
        glDrawArrays(GL_TRIANGLES, 0, self.NumVertices)

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