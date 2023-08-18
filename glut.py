import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLEW import *

# Initialize data
Frames = 100
W, H = 640, 480
data = np.random.rand(Frames, H, W, 3).astype(np.float32) * 255

textures = []
current_frame = 0


def init():
    glewInit()
    for frame_data in data:
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_FLOAT, frame_data)
        textures.append(texture)


def display():
    global current_frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBindTexture(GL_TEXTURE_2D, textures[current_frame])
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1)
    glTexCoord2f(1, 0);
    glVertex2f(1, -1)
    glTexCoord2f(1, 1);
    glVertex2f(1, 1)
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1)
    glEnd()

    glutSwapBuffers()

    current_frame = (current_frame + 1) % Frames
    glutTimerFunc(1000 // 30, lambda x: glutPostRedisplay(), 0)  # 30 FPS


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(W, H)
    glutCreateWindow("Video Player")

    init()

    glutDisplayFunc(display)
    glutTimerFunc(1000 // 30, lambda x: glutPostRedisplay(), 0)  # 30 FPS
    glutMainLoop()


if __name__ == "__main__":
    main()
