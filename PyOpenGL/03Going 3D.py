
# Python   3.4
# PySide   1.2.4
# PyOpenGL 3.1.0

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

        # Define parameters
        self.NumVAOs = 1
        self.NumVBOs = 1
        self.NumEBOs = 1

        self.uint_size = sizeof(c_uint16)
        self.float_size = sizeof(c_float)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

        # Build and compile shader program
        self.Shader = Shader()

        # set up vertex data
        self.vertices = np.array([0.5,  0.5, 0.0,
                                  0.5, -0.5, 0.0,
                                  -0.5, -0.5, 0.0,
                                  -0.5,  0.5, 0.0], dtype='float32')
        self.indices = np.array([0, 1, 3,
                                 1, 2, 3], dtype='uint16')

        self.VAO = glGenVertexArrays(self.NumVAOs)
        self.VBO = glGenBuffers(self.NumVBOs)
        self.EBO = glGenBuffers(self.NumEBOs)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 12 * self.float_size, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * self.uint_size, self.indices, GL_STATIC_DRAW)

        # position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(0)

    def paintGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Activate shader (use program)
        self.Shader.use()

        # Model matrix consists of translations, scaling and/or rotations.
        # In this case, rotate the plane on the x-axis so it looks like laying on the floor.
        model = self.getRotationMatrix(-55.0, np.array([1.0, 0.0, 0.0]))

        # View matrix, move slightly backwards in the scene so the object becomes visible
        view = self.getTranslateMatrix(np.array([0.0, 0.0, -3.0]))

        # Projection matrix. Use perspective projection (the further its vertices are, the smaller it should get)
        projection = self.getPerspectiveMatrix()

        # Retrieve the matrix uniform locations
        modelLoc = glGetUniformLocation(self.Shader.program, "model")
        viewLoc = glGetUniformLocation(self.Shader.program, "view")
        projectionLoc = glGetUniformLocation(self.Shader.program, "projection")

        # Send matrix to shaders (this is usually done in each render iteration since transformation matrices tend to change a lot)
        glUniformMatrix4fv(modelLoc,      1, GL_TRUE, model)
        glUniformMatrix4fv(viewLoc,       1, GL_TRUE, view)
        glUniformMatrix4fv(projectionLoc, 1, GL_TRUE, projection)

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)

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
        Rx, Ry, Rz = axis / np.linalg.norm(axis)  # Normalize rotation axis

        cosa = np.cos(np.deg2rad(angle))
        sina = np.sin(np.deg2rad(angle))
        sup = 1 - cosa

        rotationMatrix = np.matrix(
            [[cosa + Rx * Rx * sup, Rx * Ry * sup - Rz * sina, Rx * Rz * sup + Ry * sina, 0],
             [Ry * Rx * sup + Rz * sina,    cosa + Ry * Ry * sup, Ry * Rz * sup - Rx * sina, 0],
             [Rz * Rx * sup - Ry * sina, Rz * Ry * sup + Rx * sina,    cosa + Rz * Rz * sup, 0],
             [0,                 0,                 0, 1]], dtype='float32')

        return rotationMatrix

    def getPerspectiveMatrix(self, fov=45.0, aspect=8 / 6, near=0.1, far=100.0):
        fov_radian = np.deg2rad(fov)
        f = np.tan(fov_radian / 2)

        a11 = 1 / (f * aspect)
        a22 = 1 / f
        a33 = (near + far) / (near - far)
        a34 = 2 * near * far / (near - far)

        projectionMatrix = np.matrix([[a11, 0.0, 0.0, 0.0],
                                      [0.0, a22, 0.0, 0.0],
                                      [0.0, 0.0, a33, a34],
                                      [0.0, 0.0, -1.0, 0.0]], dtype='float32')
        return projectionMatrix


class Shader:
    def __init__(self):

        self.program = self.createProgram()

    def createProgram(self):
        # vertex shader
        strVertexShader = """
        #version 430 core
        layout (location = 0) in vec3 aPos;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
        """
        # fragment shader
        strFragmentShader = """
        #version 430 core

        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(0.1f, 1.0f, 1.0f, 1.0f);
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

        # shader program
        program = glCreateProgram()
        glAttachShader(program, vertexShader)
        glAttachShader(program, fragShader)
        glLinkProgram(program)
        self.checkCompileError(program, "Program")

        # delete the shaders as they're linked into our program and no longer necessary
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


def main():
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())


main()
