from OpenGL.GL import *; from OpenGL.GLU import *; from OpenGL.GLUT import *; import numpy as np;
import ctypes;
windowWidth = 800; windowHeight = 600;
camx = 0.0; camy = 0.0; camz = -5.0; lookx = 0.0; looky = 0.0; lookz = 5.0; upx = 0.0; upy = 1.0; upz = 0.0;
mousex = windowWidth/2; mousey = windowHeight/2;
def mouseMotion(x, y): global mousex, mousey; mousex = 0 if x < 0 else windowWidth if x > windowWidth else x; mousey = 0 if y < 0 else windowHeight if y > windowHeight else y; pass;
def mouseMouse(btn, stt, x, y): pass;
2
def paint(): # czyszczenie sceny glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
# reakcja na ruch myszką glMatrixMode(GL_MODELVIEW); glLoadIdentity(); right = np.cross(np.array([lookx, looky, lookz]), np.array([upx, upy, upz])); look = np.array([lookx, looky, lookz]); right = right / np.linalg.norm(right); up = np.cross(right, look); look -= right * 5.0 * (windowWidth/2 - mousex)/windowWidth; look -= up * 5.0 * (windowHeight/2 - mousey) / windowHeight; lookx2 = look[0]; looky2 = look[1]; lookz2 = look[2]; lookx2 = lookx2 / np.linalg.norm(look); looky2 = looky2 / np.linalg.norm(look); lookz2 = lookz2 / np.linalg.norm(look); atx = camx + lookx2; aty = camy + looky2; atz = camz + lookz2; gluLookAt(camx, camy, camz, atx, aty, atz, upx, upy, upz);
# czerwony trójkąt glColor3f(1.0, 0.0, 0.0); glBegin(GL_POLYGON); glVertex3f(-1.0, 0.0, 5.0); glVertex3f(1.0, 0.0, 5.0); glVertex3f(0.0, 1.0, 5.0); glEnd();
# zielony prostokąt glColor3f(0.0, 1.0, 0.0); glBegin(GL_QUADS); glVertex(2.0, 0.0, 0.0); glVertex(3.0, 0.0, 0.0); glVertex(3.0, 1.0, 0.0); glVertex(2.0, 1.0, 0.0); glEnd();
# niebieski wielokąt glColor3f(0.0, 0.0, 1.0); glBegin(GL_QUADS); glVertex(-3.0, 0.0, 0.0); glVertex(-2.0, 1.0, 0.0); glVertex(-3.0, 2.0, 0.0); glVertex(-4.0, 2.0, 0.0); glVertex(-4.0, 1.0, 0.0); glEnd();
# celownik
3
glColor(0.0, 0.0, 0.0); glPushMatrix(); glLoadIdentity(); gluLookAt(0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); glBegin(GL_LINES); glVertex2f(-0.2, 0.0); glVertex2f(-0.1, 0.0); glVertex2f(0.0, -0.2); glVertex2f(0.0, -0.1); glVertex2f(0.2, 0.0); glVertex2f(0.1, 0.0); glVertex2f(0.0, 0.2); glVertex2f(0.0, 0.1); glEnd(); glPopMatrix();
glutSwapBuffers(); pass;
# utworzenie okna glutInit(sys.argv); glutInitWindowPosition(int((ctypes.windll.user32.GetSystemMetrics(0) - windowWidth)/2), int((ctypes.windll.user32.GetSystemMetrics(1) - windowHeight)/2)); glutInitWindowSize(windowWidth, windowHeight); glutCreateWindow(b"PyOpenGL");
# konfiguracja opengl glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH); glutIdleFunc(paint); glutDisplayFunc(paint); glutMouseFunc(mouseMouse); glutMotionFunc(mouseMotion); glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glEnable(GL_PROGRAM_POINT_SIZE); glPointSize(5.0);
# przygotowanie sceny glClearColor(1.0, 1.0, 1.0, 0.0); glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(90.0, float(windowWidth/windowHeight), 0.1, 100.0); glMatrixMode(GL_MODELVIEW);
# pętla programu glutMainLoop();
