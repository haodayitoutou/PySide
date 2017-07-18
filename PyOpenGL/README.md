
# PyOpenGL

01/02 are examples in the book *OpenGL Programming Guide, Eighth Edition*.  
03~07 are examples from the online tutorial "Learn OpenGL"(https://learnopengl.com/).

Instead of C++ as in both the book and the online tutorial, I use PyOpenGL, the Python binding to OpenGL and releated APIs. Instead of OpenGL Utility Toolkit(glut), I use QGLWidget from PySide to configure and open windows.

00 Create a window and setup a 3D environment for rendering;  
01 A simple yet complete OpenGL program, which renders 2 blue triangles in the window;  
02 Four commands to draw a triangle;  
03 Show objects as real 3D objects by transforming 3D coordinates to 2D coordinates. Also, abstract shader into a class;  
04 Draw a 3D cube and make it rotating by changing the model matrix;  
05 Introduce camera/view space. Rotate the camera;  
06 Introduce key/mouse input to move the camera;  
07 Abstract the camera into a class. The camera's movement is far from ideal. I will stop here and work on the keybinding first then come back and try to improve it;
