"""MediaPipe Face Mesh landmark indices used by the feature modules.

Indices refer to the 468-point mesh (478 with iris refinement). "Right/left" are the
SUBJECT's right/left; the subject's right eye appears on the image's left side.
"""
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # p1 outer, p2, p3 upper, p4 inner, p5, p6 lower
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]   # p1 inner, p2, p3 upper, p4 outer, p5, p6 lower
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
MOUTH_UPPER_INNER, MOUTH_LOWER_INNER = 13, 14
MOUTH_LEFT_INNER, MOUTH_RIGHT_INNER = 78, 308
RIGHT_BROW, LEFT_BROW = 105, 334
FOREHEAD, CHIN, NOSE_TIP = 10, 152, 1
RIGHT_EYE_OUTER, LEFT_EYE_OUTER = 33, 263
MOUTH_RIGHT_CORNER, MOUTH_LEFT_CORNER = 61, 291   # image-left, image-right
N_MESH = 468
N_MESH_WITH_IRIS = 478
