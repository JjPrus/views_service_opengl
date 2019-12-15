from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time
import numpy as np

# zmienne pomocnicze
pointSize = 5
windowSize = 200
clearColor = [0.0, 0.0, 0.0]
key = 0
obj = 0
change = 1
fig = 0

pixelMapR = [[clearColor[0] for y in range(windowSize)] for x in range(windowSize)]
pixelMapG = [[clearColor[1] for y in range(windowSize)] for x in range(windowSize)]
pixelMapB = [[clearColor[2] for y in range(windowSize)] for x in range(windowSize)]


class OP:  # parametry projekcji
    l = -10
    r = 10
    b = -10
    t = 10
    n = 10
    f = 100


def clearMap(color):
    global pixelMapR, pixelMapG, pixelMapB
    for i in range(windowSize):
        for j in range(windowSize):
            pixelMapR[i][j] = color[0]
            pixelMapG[i][j] = color[1]
            pixelMapB[i][j] = color[2]


# funkcja rysująca zawartość macierzy pixelMap
def paint():
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_POINTS)
    for i in range(windowSize):
        for j in range(windowSize):
            glColor3f(pixelMapR[i][j], pixelMapG[i][j], pixelMapB[i][j])
            glVertex2f(0.5 + 1.0 * i, 0.5 + 1.0 * j)
    glEnd()
    glFlush()


# inicjalizacja okna
glutInit()
glutInitWindowSize(windowSize*pointSize, windowSize*pointSize)
glutInitWindowPosition(0, 0)
glutCreateWindow(b"Lab04")
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)

# inicjalizacja wyświetlania
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0.0, windowSize, 0.0, windowSize)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glutDisplayFunc(paint)
glutIdleFunc(paint)
glClearColor(1.0, 1.0, 1.0, 1.0)
glEnable(GL_PROGRAM_POINT_SIZE)
glPointSize(pointSize)


def cupdate(step = 0.1):
    global tick
    ltime = time.clock()
    if ltime < tick + step:
        return False
    tick = ltime
    return True


def odcinek(x1, y1, x2, y2, R, G, B): # odcinek w 2d
    global pixelMapR
    global pixelMapG
    global pixelMapB
    if x2 == x1 and y2 == y1:
        x1 = round(x1)
        y1 = round(y1)
        if 0 <= x1 < windowSize:
            if 0 <= y1 < windowSize:
                pixelMapR[x1][y1] = R
                pixelMapR[x1][y1] = G
                pixelMapR[x1][y1] = B
        return
    ony = False
    d1 = None
    d2 = None
    if x2 == x1:
        d2 = 0
    elif y2 == y1:
        d1 = 0
    else:
        d2 = (x2 - x1) / (y2 - y1)
        if not -1 < d2 < 1:
            d1 = 1 / d2
    if d1 is not None:
        d = d1
        if x1 > x2:
            xtmp = x1; x1 = x2; x2 = xtmp
            ytmp = y1; y1 = y2; y2 = ytmp
        y = y1 - d
        for x in range(int(x1), int(x2)+1):
            y = y + d
            dcx = x
            dcy = int(y)
            if 0 <= dcx < windowSize:
                if 0 <= dcy < windowSize:
                    pixelMapR[dcx][dcy] = R
                    pixelMapG[dcx][dcy] = G
                    pixelMapB[dcx][dcy] = B
    else:
        d = d2
        if y1 > y2:
            xtmp = x1; x1 = x2; x2 = xtmp
            ytmp = y1; y1 = y2; y2 = ytmp
        x = x1 - d
        for y in range(int(y1), int(y2)+1):
            x = x + d
            dcy = y
            dcx = int(x)
            if 0 <= dcx < windowSize:
                if 0 <= dcy < windowSize:
                    pixelMapR[dcx][dcy] = R
                    pixelMapG[dcx][dcy] = G
                    pixelMapB[dcx][dcy] = B


def punkt(x, y, R, G, B):  # punkt w 2d
    global pixelMapR, pixelMapG, pixelMapB
    if 0 <= x <= windowSize:
        if 0 <= y <= windowSize:
            pixelMapR[x][y] = R
            pixelMapG[x][y] = G
            pixelMapB[x][y] = B


def persp(p, l, r, b, t, n, f):  # projekcja ortograficzna
    x = p[0]
    y = p[1]
    z = p[2]

    ret = [2 * ((x * n - r * z) / (r * z - l * z)) + 1,
        2 * ((y * n - t * z) / (t * z - b * z)) + 1,
        1 - 2 * ((z - f) / (n - f))]
    return ret


def screen(p, width, height): # przekształcanie na wymiary ekranu
    ret = [(width - 1) * (p[0] + 1) / 2, (height - 1) * (p[1] + 1) / 2]
    return ret


def odcinek3D(p1, p2, R, G, B): # rysowanie odcinka w 3D
    p1o = persp(p1, OP.l, OP.r, OP.b, OP.t, OP.n, OP.f)
    p2o = persp(p2, OP.l, OP.r, OP.b, OP.t, OP.n, OP.f)
    p1s = screen([p1o[0], p1o[1]], windowSize, windowSize)
    p2s = screen([p2o[0], p2o[1]], windowSize, windowSize)
    odcinek(p1s[0], p1s[1], p2s[0], p2s[1], R, G, B)


def szes(dlugoscboku, srodek, rotx, roty, rotz, R, G, B):
    pkt = [[srodek[0] - dlugoscboku, srodek[1] - dlugoscboku, srodek[2] - dlugoscboku],
        [srodek[0] + dlugoscboku, srodek[1] - dlugoscboku, srodek[2] - dlugoscboku],
        [srodek[0] + dlugoscboku, srodek[1] + dlugoscboku, srodek[2] - dlugoscboku],
        [srodek[0] - dlugoscboku, srodek[1] + dlugoscboku, srodek[2] - dlugoscboku],
        [srodek[0] - dlugoscboku, srodek[1] - dlugoscboku, srodek[2] + dlugoscboku],
        [srodek[0] + dlugoscboku, srodek[1] - dlugoscboku, srodek[2] + dlugoscboku],
        [srodek[0] + dlugoscboku, srodek[1] + dlugoscboku, srodek[2] + dlugoscboku],
        [srodek[0] - dlugoscboku, srodek[1] + dlugoscboku, srodek[2] + dlugoscboku]]

    for i in range(8):
        npkt = np.matmul(np.array([[1, 0, 0], [0, np.cos(rotx), -np.sin(rotx)],
                                   [0, np.sin(rotx), np.cos(rotx)]]), np.array(pkt[i]).transpose())
        npkt = npkt.tolist()
        pkt[i] = npkt

    for i in range(8):
        npkt = np.matmul(np.array([[np.cos(roty), 0, np.sin(roty)], [0, 1, 0],
                                   [-np.sin(roty), 0, np.cos(roty)]]), np.array(pkt[i]).transpose())
        npkt = npkt.tolist()
        pkt[i] = npkt

    for i in range(8):
        npkt = np.matmul(np.array([[np.cos(rotz), -np.sin(rotz), 0],
                                   [np.sin(rotz), np.cos(rotz), 0], [0, 0, 1]]), np.array(pkt[i]).transpose())
        npkt = npkt.tolist()
        pkt[i] = npkt

    odcinek3D(pkt[0], pkt[1], R, G, B)
    odcinek3D(pkt[1], pkt[2], R, G, B)
    odcinek3D(pkt[2], pkt[3], R, G, B)
    odcinek3D(pkt[3], pkt[0], R, G, B)
    odcinek3D(pkt[4], pkt[5], R, G, B)
    odcinek3D(pkt[5], pkt[6], R, G, B)
    odcinek3D(pkt[6], pkt[7], R, G, B)
    odcinek3D(pkt[7], pkt[4], R, G, B)
    odcinek3D(pkt[0], pkt[4], R, G, B)
    odcinek3D(pkt[1], pkt[5], R, G, B)
    odcinek3D(pkt[2], pkt[6], R, G, B)
    odcinek3D(pkt[3], pkt[7], R, G, B)


def punkt3D(p):
    po = persp(p, OP.l, OP.r, OP.b, OP.t, OP.n, OP.f)
    ps = screen(po, windowSize, windowSize)
    punkt(round(ps[0]), round(ps[1]), 1.0, 1.0, 1.0)


def odcinek3D_w(p1, p2):
    p1o = persp(p1, OP.l, OP.r, OP.b, OP.t, OP.n, OP.f)
    p2o = persp(p2, OP.l, OP.r, OP.b, OP.t, OP.n, OP.f)
    p1s = screen(p1o, windowSize, windowSize)
    p2s = screen(p2o, windowSize, windowSize)

    odcinek(p1s[0], p1s[1], p2s[0], p2s[1], 1.0, 1.0, 1.0)


def trojkat(p1, p2, p3):
    odcinek3D_w(p1, p2)
    odcinek3D_w(p2, p3)
    odcinek3D_w(p3, p1)


def prostokat(plewygorny, pprawydolny):
    p1 = plewygorny
    p3 = pprawydolny
    p2 = [plewygorny[0], pprawydolny[1], plewygorny[2]]
    p4 = [pprawydolny[0], plewygorny[1], pprawydolny[2]]

    odcinek3D_w(p1, p2)
    odcinek3D_w(p2, p3)
    odcinek3D_w(p3, p4)
    odcinek3D_w(p1, p4)


def prostopadloscian(dlugoscbokuA, dlugoscbokuB, dlugoscbokuC, psrodek):
    dlugoscbokuA = dlugoscbokuA / 2
    dlugoscbokuB = dlugoscbokuB / 2
    dlugoscbokuC = dlugoscbokuC / 2

    p1 = [psrodek[0] - dlugoscbokuA, psrodek[1] - dlugoscbokuB, psrodek[2] - dlugoscbokuC]
    p2 = [psrodek[0] + dlugoscbokuA, psrodek[1] - dlugoscbokuB, psrodek[2] - dlugoscbokuC]
    p3 = [psrodek[0] + dlugoscbokuA, psrodek[1] + dlugoscbokuB, psrodek[2] - dlugoscbokuC]
    p4 = [psrodek[0] - dlugoscbokuA, psrodek[1] + dlugoscbokuB, psrodek[2] - dlugoscbokuC]
    p5 = [psrodek[0] - dlugoscbokuA, psrodek[1] - dlugoscbokuB, psrodek[2] + dlugoscbokuC]
    p6 = [psrodek[0] + dlugoscbokuA, psrodek[1] - dlugoscbokuB, psrodek[2] + dlugoscbokuC]
    p7 = [psrodek[0] + dlugoscbokuA, psrodek[1] + dlugoscbokuB, psrodek[2] + dlugoscbokuC]
    p8 = [psrodek[0] - dlugoscbokuA, psrodek[1] + dlugoscbokuB, psrodek[2] + dlugoscbokuC]

    odcinek3D_w(p1, p2)
    odcinek3D_w(p2, p3)
    odcinek3D_w(p3, p4)
    odcinek3D_w(p1, p4)
    odcinek3D_w(p5, p6)
    odcinek3D_w(p6, p7)
    odcinek3D_w(p7, p8)
    odcinek3D_w(p5, p8)
    odcinek3D_w(p1, p5)
    odcinek3D_w(p2, p6)
    odcinek3D_w(p3, p7)
    odcinek3D_w(p4, p8)


def szescian(dlugoscboku, psrodek, p0, v, phi):
    p = np.array([[psrodek[0] - dlugoscboku, psrodek[1] - dlugoscboku, psrodek[2] - dlugoscboku],
        [psrodek[0] + dlugoscboku, psrodek[1] - dlugoscboku, psrodek[2] - dlugoscboku],
        [psrodek[0] + dlugoscboku, psrodek[1] + dlugoscboku, psrodek[2] - dlugoscboku],
        [psrodek[0] - dlugoscboku, psrodek[1] + dlugoscboku, psrodek[2] - dlugoscboku],
        [psrodek[0] - dlugoscboku, psrodek[1] - dlugoscboku, psrodek[2] + dlugoscboku],
        [psrodek[0] + dlugoscboku, psrodek[1] - dlugoscboku, psrodek[2] + dlugoscboku],
        [psrodek[0] + dlugoscboku, psrodek[1] + dlugoscboku, psrodek[2] + dlugoscboku],
        [psrodek[0] - dlugoscboku, psrodek[1] + dlugoscboku, psrodek[2] + dlugoscboku]])

    v_len = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if v_len != 1:
        v = v / v_len

    a = v[0]
    b = v[1]
    c = v[2]
    M = np.array([[a ** 2 * (1 - np.cos(phi)) + np.cos(phi), a * b * (1 - np.cos(phi)) - c * np.sin(phi),
                   a * c * (1 - np.cos(phi)) + b * np.sin(phi)],
                  [a * b * (1 - np.cos(phi)) + c * np.sin(phi), b ** 2 * (1 - np.cos(phi)) + np.cos(phi),
                   b * c * (1 - np.cos(phi)) - a * np.sin(phi)],
                  [a * c * (1 - np.cos(phi)) - b * np.sin(phi), b * c * (1 - np.cos(phi)) + a * np.sin(phi),
                   c ** 2 * (1 - np.cos(phi)) + np.cos(phi)]])

    for i in range(len(p)):
        p[i] = p[i] - p0
        p[i] = M @ p[i].T
        p[i] = p[i].T
        p[i] = p[i] + np.array(p0)

    odcinek3D_w(p[0], p[1])
    odcinek3D_w(p[1], p[2])
    odcinek3D_w(p[2], p[3])
    odcinek3D_w(p[0], p[3])
    odcinek3D_w(p[4], p[5])
    odcinek3D_w(p[5], p[6])
    odcinek3D_w(p[6], p[7])
    odcinek3D_w(p[4], p[7])
    odcinek3D_w(p[0], p[4])
    odcinek3D_w(p[1], p[5])
    odcinek3D_w(p[2], p[6])
    odcinek3D_w(p[3], p[7])


def keyboard(k, x, y):
    global key
    key = k.decode("utf-8")


while True:
    clearMap([0.0, 0.0, 0.0])
    glutKeyboardFunc(keyboard)

    if key == '1':
        fig = 1
    elif key == '2':
        fig = 2
    elif key == '3':
        fig = 3
    elif key == '4':
        fig = 4
    elif key == '5':
        fig = 5
    elif key == 'q':
        OP.l -= change
        key = 0
    elif key == 'w' and OP.l < OP.r:
        OP.l += change
        key = 0
    elif key == 'e' and OP.r > OP.l:
        OP.r -= change
        key = 0
    elif key == 'r':
        OP.r += change
        key = 0
    elif key == 'z':
        OP.b -= change
        key = 0
    elif key == 'a' and OP.b < OP.t:
        OP.b += change
        key = 0
    elif key == 'x' and OP.t > OP.b:
        OP.t -= change
        key = 0
    elif key == 's':
        OP.t += change
        key = 0
    elif key == 'u' and OP.n > 0:
        OP.n -= change
        key = 0
    elif key == 'i' and OP.n < OP.f:
        OP.n += change
        key = 0
    elif key == 'o' and OP.f > OP.n:
        OP.f -= change
        key = 0
    elif key == 'p':
        OP.f += change
        key = 0

    if fig == 1:
        punkt3D([2, 2, 30])
    elif fig == 2:
        odcinek3D_w([2, 2, 30], [10, 10, 40])
    elif fig == 3:
        trojkat([2, 2, 30], [8, 8, 40], [8, 2, 50])
    elif fig == 4:
        prostokat([2, 2, 30], [8, 10, 40])
    elif fig == 5:
        prostopadloscian(1, 1, 2, [1, 0, 3])

    paint()
    glutMainLoopEvent()
