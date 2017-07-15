
# Python   3.4
# PySide   1.2.4
# PyOpenGL 3.1.0

import sys
import numpy as np
import time
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

        #Define parameters
        self.NumVAOs = 1
        self.NumVBOs = 1
        self.NumEBOs = 1
        
        self.uint_size = sizeof(c_uint16)
        self.float_size = sizeof(c_float)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

        #Build and compile shader program
        self.Shader = Shader()

        #set up vertex data. 8 vertices
        x = 0.5
        self.vertex_pos = x * np.array([
            -1, -1, -1,
             1, -1, -1,
             1,  1, -1,
            -1,  1, -1,

            -1,  1,  1,
            -1, -1,  1,
             1, -1,  1,
             1,  1,  1], dtype='float32')

        #8 colors, one for each vertex
        self.vertex_col = np.array([
            1.0, 1.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0], dtype='float32')

        self.indices = np.array([
            0, 1, 2,
            0, 2, 3,
            5, 0, 3,
            5, 3, 4,
            6, 5, 4,
            6, 4, 7,
            1, 6, 7,
            1, 7, 2,
            3, 2, 7,
            3, 7, 4,
            5, 6, 1,
            5, 1, 0], dtype='uint16') #12 triangles, 6 faces
        
        sizePos = self.float_size * self.vertex_pos.size
        sizeCol = self.float_size * self.vertex_col.size
        sizeIdx = self.uint_size * self.indices.size

        self.VAO = glGenVertexArrays(self.NumVAOs)
        self.VBO = glGenBuffers(self.NumVBOs)
        self.EBO = glGenBuffers(self.NumEBOs)

        #Initialize vertex array object (VAO)
        glBindVertexArray(self.VAO)
        
        #Allocate vertex buffer object (VBO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, sizePos+sizeCol, None, GL_STATIC_DRAW) #allocate space for VBO (pos+col)
        glBufferSubData(GL_ARRAY_BUFFER,       0, sizePos, self.vertex_pos)  #"pos" at offset zero in the buffer
        glBufferSubData(GL_ARRAY_BUFFER, sizePos, sizeCol, self.vertex_col)  #"col" after "pos". Offset = sizePos

        #Allocate element buffer object (EBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeIdx, self.indices, GL_STATIC_DRAW)

        #Associate variables in a vertex shader with data stored in VBO
        stride_vertex = 3 * self.float_size
        stride_color  = 3 * self.float_size
        offset_vertex = c_void_p(0)
        offset_color  = c_void_p(sizePos)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride_vertex, offset_vertex)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride_color, offset_color)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

    def paintGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Activate shader (use program)
        self.Shader.use()
        
        #Model matrix. Change the angle so that it will rotate
        angle = time.time() * 50
        axis = np.array([1.0, 0.3, -0.5]) #rotating axis
        model = self.getRotationMatrix(angle, axis)
        self.Shader.setUniformMatrix("model", model)

        #View matrix, move slightly backwards in the scene so the object becomes visible
        view = self.getTranslateMatrix(np.array([0.0, 0.0, -5.0]))
        self.Shader.setUniformMatrix("view", view)

        #Projection matrix. Use perspective projection (the further its vertices are, the smaller it should get)
        projection = self.getPerspectiveMatrix()
        self.Shader.setUniformMatrix("projection", projection)

        #Render cubes
        glBindVertexArray(self.VAO)

        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, None)

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

    def getTranslateMatrix(self, vector):
        x, y, z = vector
        translateMatrix = np.matrix([[1.0, 0.0, 0.0,   x],
                                     [0.0, 1.0, 0.0,   y],
                                     [0.0, 0.0, 1.0,   z],
                                     [0.0, 0.0, 0.0, 1.0]], dtype='float32')
        return translateMatrix

    def getRotationMatrix(self, angle, axis):
        Rx, Ry, Rz = axis / np.linalg.norm(axis) #Normalize rotation axis

        cosa = np.cos( np.deg2rad(angle) )
        sina = np.sin( np.deg2rad(angle) )
        sup  = 1 - cosa

        rotationMatrix = np.matrix(
            [[   cosa+Rx*Rx*sup, Rx*Ry*sup-Rz*sina, Rx*Rz*sup+Ry*sina, 0],
             [Ry*Rx*sup+Rz*sina,    cosa+Ry*Ry*sup, Ry*Rz*sup-Rx*sina, 0],
             [Rz*Rx*sup-Ry*sina, Rz*Ry*sup+Rx*sina,    cosa+Rz*Rz*sup, 0],
             [                0,                 0,                 0, 1]], dtype='float32')

        return rotationMatrix

    def getPerspectiveMatrix(self, fov=45.0, aspect=8/6, near=0.1, far=100.0):
        fov_radian = np.deg2rad(fov)
        f = np.tan(fov_radian/2)

        a11 = 1 / (f*aspect)
        a22 = 1 / f
        a33 = (near+far) / (near-far)
        a34 = 2 * near * far / (near-far)

        projectionMatrix = np.matrix([[a11, 0.0, 0.0, 0.0],
                                      [0.0, a22, 0.0, 0.0],
                                      [0.0, 0.0, a33, a34],
                                      [0.0, 0.0,-1.0, 0.0]], dtype='float32')
        return projectionMatrix


class Shader:
    def __init__(self):

        self.program = self.createProgram()

    def createProgram(self):
        #vertex shader
        strVertexShader = """
        #version 430 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aCol;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 outColor;
        
        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            outColor = aCol;
        }
        """
        #fragment shader
        strFragmentShader = """
        #version 430 core

        in vec3 outColor;
        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(outColor, 1.0);
        }
        #"""

        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, strVertexShader)
        glCompileShader(vertexShader)
        self.checkCompileError(vertexShader, "vertex")

        fragShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragShader, strFragmentShader)
        glCompileShader(fragShader)
        self.checkCompileError(fragShader, "fragment")

        #shader program
        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragShader)
        glLinkProgram(program)
        self.checkCompileError(program, "Program")

        #delete the shaders as they're linked into our program and no longer necessary
        glDeleteShader(vertexShader)
        glDeleteShader(fragShader)

        return program

    def use(self):
        glUseProgram(self.program)

    def checkCompileError(self, shader, type):
        if type != "Program":
            if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
                raise ValueError("Compile error in the {} shader.".format(type))
        else:
            if glGetProgramiv(shader, GL_LINK_STATUS) != GL_TRUE:
                raise ValueError("Compile error in the program.")

    def setUniformMatrix(self, uName, uValue):
        uniformLoc = glGetUniformLocation(self.program, uName)
        glUniformMatrix4fv(uniformLoc, 1, GL_TRUE, uValue)


def main():
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    #Set a time to trigger the re-rendering every 0.02 sec
    timer = QTimer()
    timer.timeout.connect( window.glWidget.updateGL )
    timer.start(20)

    sys.exit(app.exec_())
    
main()
