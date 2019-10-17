from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

def rectContains(rect, point) :
    #if point[0] < rect[0] :
    #    return False
    #elif point[1] < rect[1] :
    #    return False
    #elif point[0] > rect[2] :
    #    return False
    #elif point[1] > rect[3] :
    #    return False
    #return True
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True
    
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))
    #subdiv.insert(ctr)
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri

def warpTriangles(treug_old, treug_new, image_file):
    # Read input image and convert to float
    img1 = cv2.imread(image_file)
    img2 = 0 * np.ones(img1.shape, dtype = img1.dtype)

    # Define input and output triangles 
    triug1 = np.float32([treug_old])
    triug2 = np.float32([treug_new])
    
    r1 = cv2.boundingRect(triug1)
    r2 = cv2.boundingRect(triug2)
    
    # Offset points by left top corner of the 
    # respective rectangles
    
    triug1Cropped = []
    triug2Cropped = []
     
    for i in range(3):
        triug1Cropped.append(((triug1[0][i][0] - r1[0]),(triug1[0][i][1] - r1[1])))
        triug2Cropped.append(((triug2[0][i][0] - r2[0]),(triug2[0][i][1] - r2[1])))

    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(triug1Cropped), np.float32(triug2Cropped) )
    
    # Apply the Affine Transform just found to the src image
    img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    
    # Get mask by filling triangle
    mask_example = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask_example, np.int32(triug2Cropped), (1.0, 1.0, 1.0), 16, 0);

    # Apply mask to cropped region
    img2Cropped = img2Cropped * mask_example

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask_example )

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped
    
    return(img2)
    
def pointTraversalOrder(points):    
    contours = np.array(points)
    rect = cv2.boundingRect(contours)
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    order = calculateDelaunayTriangles(rect, contours) # вершины, номера точек из points
    return order

def getTriangles(points, order):
    tiangles = [] # верши, пары точек
    for t in order:
        tmp1 = points[t[0]]
        tmp2 = points[t[1]]
        tmp3 = points[t[2]]
        tiangles.append([tmp1, tmp2, tmp3])
    tiangles1 = np.array(tiangles).reshape((-1,1,2)).astype(np.int32)
    tiangles2 = np.array(tiangles).astype(np.int32)
    return tiangles2

def paint(old_image_file, new_image_file, old_triangles, new_triangles):    
    img1 = cv2.imread(new_image_file)
    img2 = 0 * np.ones(img1.shape, dtype = img1.dtype)
    imgmas = []
    for i in range(len(old_triangles)):
        img2 += warpTriangles(old_triangles[i], new_triangles[i], old_image_file)
        imgmas.append(warpTriangles(old_triangles[i], new_triangles[i], old_image_file))
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()