import shelve
import numpy as np
import cv2
import time

def get_video(pred):
    with shelve.open('default_points.txt', 'r') as db:
        default_points = db['default_face']
        points_to_indexes = {tuple(p): i for i, p in enumerate(default_points)}

    default_face = cv2.imread('default_image.jpg')
    subdiv = cv2.Subdiv2D((0,0,320,320))

    for p in default_points:
        subdiv.insert(tuple(p))

    def get_frame(delta):
        # delta: list of ### with shape (num_points, 2)
        triangles = []

        for t in subdiv.getTriangleList():
            
            if not {tuple(t[i: i + 2]) for i in (0, 2, 4)}.issubset(set(points_to_indexes.keys())):
                continue

            # affine transform
            old_pts = np.array([t[:2], t[2:4], t[4:]], np.int32)
            d = np.array([delta[points_to_indexes[tuple(t[i: i + 2])]]  for i in (0, 2, 4)])
            print(d)
            new_pts = old_pts + d
            affinne_transform = cv2.getAffineTransform(old_pts.astype(np.float32), new_pts.astype(np.float32))
            aff_image = cv2.warpAffine(default_face, affinne_transform, default_face.shape[:2], None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # masking of affine image
            mask = np.zeros(aff_image.shape, dtype=np.uint8)
            channel_count = aff_image.shape[2]
            ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(mask, [new_pts], ignore_mask_color)
            masked_image = cv2.bitwise_and(aff_image, mask)

            triangles.append(masked_image)

        triangles = np.array(triangles, dtype=np.float32)
        #print('triangles: ', triangles.shape)

        t1 = time.time()
        #print('triangles: ', triangles.shape)
        triangles[(triangles == 0).all(axis=3)] = np.nan
        res = np.nan_to_num(np.nanmean(triangles, axis=0)).astype(np.uint8)

        #print(time.time() - t1)
        return res

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 25, (320,320))
    
    for i, delta in enumerate(pred):
        print('Frame number {} : {}'.format(i + 1, len(pred)))
        out.write(get_frame(delta))
    
    out.release()
