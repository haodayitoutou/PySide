
#Python   3.4
#PySide   1.2.4
#PyOpenGL 3.1.0

import sys
import numpy as np
from ctypes import sizeof, c_uint16, c_float, c_void_p

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
        self.NumVBOs = 1
        self.NumEBOs = 1

        self.uint_size   = sizeof(c_uint16)
        self.float_size = sizeof(c_float)
        
    def initializeGL(self):
        #Vertices
        x = 1.0
        self.vertex_pos = np.array([-x, -x, 0.0, 1.0,
                                     x, -x, 0.0, 1.0,
                                    -x,  x, 0.0, 1.0,
                                     x,  x, 0.0, 1.0], dtype='float32')
            #The last vertex is used to demonstrate glDrawElementsBaseVertex(),
            #   which has an offset of 1 and will draw the last 3 vertices in the last 3 colors.
            #I change its coordinate from (-x,-x) to (x,x).
            #In the book, the only difference is the color, as the 4th vertex is the same as the first one.
            #With (x,x), it will be more obvious, with differences in both color and position.
        self.vertex_col = np.array([1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 0.0, 1.0,
                                    1.0, 0.0, 1.0, 1.0,
                                    0.0, 1.0, 1.0, 1.0], dtype='float32')
        self.indices = np.array([0, 1, 2], dtype='uint16')

        sizePos = self.float_size * 16
        sizeCol = self.float_size * 16
        sizeIdx = self.uint_size * 3

        #Initialize vertex array object (VAO)
        self.VAOs = glGenVertexArrays(self.NumVAOs)
        glBindVertexArray(self.VAOs)

        #Allocate vertex buffer object (VBO)
        self.VBOs = glGenBuffers(self.NumVBOs)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs)

        glBufferData(GL_ARRAY_BUFFER, sizePos+sizeCol, None, GL_STATIC_DRAW) #allocate space for VBO (pos+col)
        glBufferSubData(GL_ARRAY_BUFFER,       0, sizePos, self.vertex_pos)  #"pos" at offset zero in the buffer
        glBufferSubData(GL_ARRAY_BUFFER, sizePos, sizeCol, self.vertex_col)  #"col" after "pos". Offset = sizePos

        #Allocate element buffer object (EBO)
        self.EBOs = glGenBuffers(self.NumEBOs)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBOs)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeIdx, self.indices, GL_STATIC_DRAW)

        #Associate variables in a vertex shader with data stored in VBO
        stride_pos = 4 * self.float_size
        stride_col = 4 * self.float_size #4 floats for each vertex
        offset_pos = c_void_p(0)
        offset_col = c_void_p(sizePos) #skip the whole "pos" values. Offset = sizePos

        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride_pos, offset_pos)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride_col, offset_col)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)


        #Initialize vertex and fragment shaders
        strVertexShader = """
        #version 430 core

        uniform mat4 model;
        uniform mat4 projection;

        layout(location=0) in vec4 vPos;
        layout(location=1) in vec4 vCol;

        out vec4 color;

        void main() {
            gl_Position = projection * model * vPos;
            color = vCol;
        }
        """
        strFragmentShader = """
        #version 430 core

        in vec4 color;
        out vec4 fColor;

        void main() {
            fColor = color;
        }
        """
        self.program = compileProgram(
            compileShader(strVertexShader, GL_VERTEX_SHADER),
            compileShader(strFragmentShader, GL_FRAGMENT_SHADER))

        #Calculate the "projection" uniform matrix
        glUseProgram(self.program)

        #Perspective projection -> glm::perspective(45.0, 800/600, 0.1,  100.0)
        fov = np.deg2rad(45)
        f = np.tan(fov/2)
        a11 = 1 / (f*8/6)
        a22 = 1 / f
        a33 = (0.1+100.0)/(0.1-100.0)
        a34 = 2*0.1*100.0/(0.1-100.0)

        projectionMatrix = np.matrix([[a11, 0.0, 0.0, 0.0],
                                      [0.0, a22, 0.0, 0.0],
                                      [0.0, 0.0, a33, a34],
                                      [0.0, 0.0,-1.0, 0.0]], dtype='float32')
        projectionLocation = glGetUniformLocation(self.program, "projection") #get the location of a uniform variable
        glUniformMatrix4fv(projectionLocation, 1, GL_TRUE, projectionMatrix) #specify the value of a uniform variable

        glUseProgram(0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        glBindVertexArray(self.VAOs) #select the vertex array as vertex data

        #Calculate the "model" uniform matrix
        modelLocation = glGetUniformLocation(self.program, "model") #get its location

        #4 triangles are drawn using 4 methods. Use modelMatrix to set their positions.

        #1 constructs a sequence of geometric primitives using array elements.
        modelMatrix = np.matrix([[1.0, 0.0, 0.0, -3.0],
                                 [0.0, 1.0, 0.0,  0.0],
                                 [0.0, 0.0, 1.0,-10.0],
                                 [0.0, 0.0, 0.0,  1.0]], dtype='float32')
        glUniformMatrix4fv(modelLocation, 1, GL_TRUE, modelMatrix)

        glDrawArrays(GL_TRIANGLES, 0, 3)

        #2 defines a sequence using 3 elements, whose indices are stored in GL_ELEMENT_ARRAY_BUFFER (EBO)
        modelMatrix[0, 3] = -1.0
        glUniformMatrix4fv(modelLocation, 1, GL_TRUE, modelMatrix)

        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, None) #SHORT - uint16, INT - uint32

        #3 This function allows the indices in EBO to be offset by a fixed amount
        modelMatrix[0, 3] = 1.0
        glUniformMatrix4fv(modelLocation, 1, GL_TRUE, modelMatrix)

        glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, None, 1) #offset=1, drawing the last 3 vertices

        #4 This function executes the same drawing commands many times in a row.
        modelMatrix[0, 3] = 3.0
        glUniformMatrix4fv(modelLocation, 1, GL_TRUE, modelMatrix)

        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 1) #last paramenter is the the count

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