import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from scipy import ndimage
import seaborn as sns
import colorsys

from imantics import Polygons, Mask
from shapely.geometry import Polygon

from random import randint

from skimage.metrics import structural_similarity as compare_ssim
import imutils

import winsound

#
# B e e p
#
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
def Beep():
    winsound.PlaySound("beep.wav", winsound.SND_FILENAME)


def load_img():
    blank_img = np.zeros((200, 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='BARAK', org=(25, 150), fontFace=font, fontScale=3, color=(255, 255, 255), thickness=16,
                lineType=cv2.LINE_AA)
    return blank_img

kernel = np.ones((10, 10), np.uint8)
def read_poly(file):
    img = Image.open(file)
    img_arr = asarray(img)
    cv2.imshow('poly-orig', img_arr)

    bwd_img = ndimage.distance_transform_edt(1- img_arr)
    cv2.imshow('BW distance orig', bwd_img)

    kernel = np.ones((5, 5), np.uint8)

    # dilation 10
    dilation10 = cv2.dilate(img_arr, kernel, iterations=10)
    cv2.imshow('dilation-10', dilation10)

#computes the mean structural similarity index between two images
    print(f"Type img_arr = {type(img_arr)}")
    print(f"Type dilation10 = {type(dilation10)}")
    grayA = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(dilation10, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    cv2.imshow("Diff-ssim 10", diff)

#    img_sub_10 = img_arr - dilation10
#    cv2.imshow('Sub orig from dialation-10', img_sub_10)

    bwd_dilation10 = ndimage.distance_transform_edt(1- dilation10)
    cv2.imshow('BW distance dialation-10', bwd_dilation10)

    threshold = 0.1
    absdiff10 = cv2.absdiff(img_arr, dilation10)
    _, thresholded = cv2.threshold(
        absdiff10, int(threshold * 255),
        255, cv2.THRESH_BINARY)
    cv2.imshow('Abs Diff 10', absdiff10)

    # dilation 3
    dilation3 = cv2.dilate(img_arr, kernel, iterations=3)
    cv2.imshow('dilation-3', dilation3)

    absdiff3 = cv2.absdiff(img_arr, dilation3)
    _, thresholded = cv2.threshold(
        absdiff3, int(threshold * 255),
        255, cv2.THRESH_BINARY)
    cv2.imshow('Abs Diff 3', absdiff3)

    grayA = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(dilation3, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    cv2.imshow("Diff-ssim 3", diff)

#    erosion3 = cv2.erode(dilation3, kernel, iterations=3)
#    cv2.imshow('dilation-erosion-3', erosion3)

    # Erosion
#    erosion1 = cv2.erode(img_arr, kernel, iterations=1)
#    cv2.imshow('erosion1', erosion1)

    # dilation
#    dilation_iter_1 = cv2.dilate(erosion1, kernel, iterations=1)
#    cv2.imshow('dilation 1 on erros 1 iter', dilation_iter_1)שששש

#    erosion3 = cv2.erode(img_arr, kernel, iterations=3)
#    cv2.imshow('erosion3', erosion3)

    # dilation
#    dilation_iter_3 = cv2.dilate(erosion3, kernel, iterations=3)
#    cv2.imshow('dilation 3 on eros 3 iter', dilation_iter_3)


#    erosion10 = cv2.erode(img_arr, kernel, iterations=10)
#    cv2.imshow('erosion10', erosion10)

    # dilation
#    dilation_iter_10 = cv2.dilate(erosion10, kernel, iterations=10)
#    cv2.imshow('dilation 10 on eros 10 iter', dilation_iter_10)


def test_morph():
    orig = load_img()
    cv2.imshow('1-orig', orig)

    kernel = np.ones((5, 5), np.uint8)

    # Erosion
    erosion1 = cv2.erode(orig, kernel, iterations=1)
    cv2.imshow('2-erosion1', erosion1)

    erosion3 = cv2.erode(orig, kernel, iterations=3)
    cv2.imshow('3-erosion3', erosion3)

    # dilation
    dilation = cv2.dilate(orig, kernel, iterations=1)
    cv2.imshow('4-dilation', dilation)

    # opening
    white_noise = np.random.randint(low=0, high=2, size=(300, 400))
    white_noise = white_noise * 255
    noise_img = white_noise + orig
    cv2.imshow('5-noise_img', noise_img)
    opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('6-opening', opening)

    # closing
    black_noise = np.random.randint(low=0, high=2, size=(300, 400))
    black_noise = black_noise * -255
    black_noise_img = orig + black_noise
    black_noise_img[black_noise_img == -255] = 0
    cv2.imshow('7-black_noise_img', black_noise_img)
    closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('8-closing', closing)

    # Morphological gradient
    gradient = cv2.morphologyEx(orig, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('9-gradient', gradient)
#
#
#

img_w = 600
img_h = 600

img = np.zeros((img_w, img_w), dtype='uint8')
imgp = np.zeros((img_w, img_w,3), dtype='uint8')
imgd = np.zeros((img_w, img_w), dtype='uint8')
img_absdiff = np.zeros((img_w, img_w), dtype='uint8')
img_bwd = np.zeros((img_w, img_w), dtype='uint8')


conturs = np.zeros((img_h, img_w, 3), dtype=np.uint8)

clone = img.copy()
temp = img.copy()

def negative(im):
    height=len(im)
    width = len(im[0])
    for row in range(height):
        for col in range(width):
            red = im[row][col][0] - 255
            green = im[row][col][1] - 255
            blue = im[row][col][2] - 255
            im[row][col]=[red,green,blue]
    return im

def dilate (img, iter=10):
    global imgd
    print(f"Dilate {iter}")
    imgd = cv2.dilate(img, kernel, iter)
    cv2.imshow('dilation', imgd)


threshold = 0.1
def absdiff(img1, img2):
    print(f"AbsDiff")
    img_absdiff = cv2.absdiff(img1, img2)
    _, thresholded = cv2.threshold(
        img_absdiff, int(threshold * 255),
        255, cv2.THRESH_BINARY)
    cv2.imshow('Abs Diff ', img_absdiff)

def bwdist(img):
    global img_bwd
    print(f"bwdist")
    img_bwd = ndimage.distance_transform_edt(1 - asarray(img))
    cv2.imshow('BW distance ', img_bwd)

def try_contour(file):
    img = cv2.imread(file)
    print(f"im type {type(img)}")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"gray type {type(img)}")
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours = " + str(len(contours)))

    cv2.drawContours(img, contours, -1, (0, 255, 0), thikness=3)
    cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Image', img)

#
# Distinct colors
#

colors = []
colors.append([0,0,255])
colors.append([0,255,0])
colors.append([255,0,0])

colors.append([255,255,0])
colors.append([255,0,255])
colors.append([0,255,255])


def gen_distinct_color_tbl_01():
    global colors
    for r in range(10,255, 64):
        for g in range(10,255, 64):
            for b in range (10,255, 64):
                colors.append([r, g, b])

def gen_distinct_color_tbl_02():
    print(f"gen_distinct_color_tbl_02")
    global colors
    clrs = sns.color_palette()

    for clr in clrs:
        rgb=[]
        for value in clr:
            value *= 255
            rgb.append((value))
        colors.append(rgb)

def show_color_palette():
    width=20
    img_palt = np.zeros((width, len(colors)*20, 3), dtype=np.uint8)

    x=0
    for clr in colors:
        cv2.rectangle(img_palt, (x, 0), (x+width, width), clr, -1)
        x=  x + 20

    cv2.imshow('Colors Palette', img_palt)


clr_idx=0
def get_contourt_from_n_draw_on_imag(img_cntr, img, new = True):
    global conturs, clr_idx
    print(f"get_contourt_from_n_draw_on_imag")
    if (new):
        conturs.fill(255)

    im = np.zeros((img_w, img_w), dtype='uint8')

    contours, hierarchy = cv2.findContours(img_cntr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    calc_accuracy_contour(contours)

    cv2.drawContours(im, contours, -1, (255, 0, 0), thickness=3)
    cv2.imshow('Contour', im)

    cv2.drawContours(asarray(conturs), contours, -1, colors[clr_idx], thickness=3)
    #    cv2.polylines(conturs, asarray(ptr_contours), True, color=colors[clr_idx], thickness=3)
    cv2.imshow('Contours', conturs)
    clr_idx = clr_idx + 1
    if clr_idx > len(colors) - 1:
        clr_idx =0;

def calc_accuracy_contour(cntrs):
    print(f"==> calc_accuracy_contour")
    img_tmp = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for c in cntrs:
        accuracy = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, accuracy, True)
        print(f" # edges = {len(approx)}")
        cv2.drawContours(img_tmp, [approx], 0, colors[clr_idx], 2)
        cv2.imshow('Accuracu poly', img_tmp)

#
#  Calculate polygon
#
def to_poly(im):
    polygons = Mask(im).polygons()
    print(f" points {len(polygons.points)}\n {polygons.points}")
    print(f" poly- segment \n {polygons.segmentation}")

def _DFS(polygons, contours, hierarchy, sibling_id, is_outer, siblings):
  while sibling_id != -1:
    contour = contours[sibling_id].squeeze(axis=1)
    if len(contour) >= 3:
      first_child_id = hierarchy[sibling_id][2]
      children = [] if is_outer else None
      _DFS(polygons, contours, hierarchy, first_child_id, not is_outer, children)

      if is_outer:
        polygon = Polygon(contour, holes=children)
        polygons.append(polygon)
      else:
        siblings.append(contour)

    sibling_id = hierarchy[sibling_id][0]

def to_poly2(im):
    print(f"==> to_poly_2")
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    contour_to_poly(contours, hierarchy)

def contour_to_poly(contours, hierarchy):
    print(f"==> contour_to_poly")
    hierarchy = hierarchy[0]
    polygons = []
    _DFS(polygons, contours, hierarchy, 0, True, [])

    print(f"contour_to_poly - #points = {len(polygons)}")
    print(f" contour_to_poly \n {polygons[0]}")
#
# ***** G U I *****
#
once = True
done = False
poly_done = False
points = []
current = (0, 0)
prev_current = (0,0)

def draw_poly():
    global points, imgp
    pts = np.array(points)
    cv2.polylines(imgp, [pts], True, (0, 0, 0), thickness=3)
    cv2.imshow('Polygon', imgp)

def init():
    global img, clone,temp, imgd, imgp, conturs
    img.fill(255)
    imgp.fill(255)
    cv2.imshow('Polygon', imgp)
    clone.fill(255)
    temp.fill(255)
    imgd.fill(255)
    img_absdiff.fill(255)
    img_bwd.fill(255)
    conturs.fill(255)
    cv2.imshow('Contours', conturs)
    cv2.imshow('Contour', img)

def gui():
    global img, done, poly_done, points, current,temp, colors_tbl
    init()
    gen_distinct_color_tbl_02()
    show_color_palette()


    def on_mouse(event, x, y, buttons, user_param):
        global img, done, poly_done, points, current, temp, once
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        if done:  # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            if (not poly_done):
                current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            if (not poly_done):
                print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
                cv2.circle(img, (x, y), 5, (0, 200, 0), -1)
                points.append([x, y])
                temp = img.copy()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            if (not poly_done):
                cv2.circle(img, (x, y), 5, (0, 200, 0), -1)
                points.append([x, y])
                temp = img.copy()
                print("Completing polygon with %d points." % len(points))
            # of a filled polygon
                poly_done = True
                img = clone.copy()
                draw_poly()
                if (len(points) > 0):
                    cv2.fillPoly(img, np.array([points]), (0, 0, 0))
                    temp = img.copy()
                    if (False):
                        once = False
                        dilate(img)
                        absdiff(img, imgd)
                        get_contourt_from_n_draw_on_imag(imgd, img)

    cv2.namedWindow("Poly-image")
    cv2.setMouseCallback("Poly-image", on_mouse)

    while (not done):
        # This is our drawing loop, we just continuously draw new images
        # and show them in the named window
        if (len(points) > 1):
            if ((current != prev_current) ):
                img = temp.copy()
            # Draw all the current polygon segments
            cv2.polylines(img, [np.array(points)], False, (0, 0, 0), 1)
            # And  also show what the current segment would look like
            cv2.line(img, (points[-1][0], points[-1][1]), current, (0, 0, 0))

        # Update the window
        cv2.imshow("Poly-image", img)
        # And wait 50ms before next iteration (this will pump window messages meanwhile)
        key = cv2.waitKey(50)
        if key == ord('q'):  # press d(done)
            done = True
        elif key == ord('d'):  # dialte + diff
            print(f"Do D ....")
            dilate(img)
            absdiff(img, imgd)
            get_contourt_from_n_draw_on_imag(imgd, img)
            #bwdist(img)
        elif key == ord('a'):  # Do 'd' again
            print(f"Do D ....")
            img = imgd.copy()
            dilate(img)
            absdiff(img, imgd)
            get_contourt_from_n_draw_on_imag(imgd, img, new=False)
        elif key == ord('n'):  # new poly
            print(f"New ....")
            poly_done = False
            temp = np.zeros((img_w, img_w))

            img = temp.copy()
            points.clear()
            init()
            cv2.imshow('dilation', imgd)
            cv2.imshow('Abs Diff ', img_absdiff)
            cv2.imshow("Poly-image", img)

def run():
#   read_poly('poly_01.jpg')
    #read_poly('bomb_ fill_hole.jpg')
    #try_contour('poly_01.jpg')
    gui()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()