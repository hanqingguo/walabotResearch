import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import PIL
import io





frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

X = [[1,2],[3,4],[5,6]]
X = np.array(X)
plt.imshow(X, cmap=plt.cm.hot)

im._rgba_cache


buffer_ = io.BytesIO()
plt.savefig( buffer_, format = "png", bbox_inches = 'tight', pad_inches = 0 )
buffer_.seek(0)

image = PIL.Image.open( buffer_ )

ar = np.asarray(image)

print(ar, ar.shape)
cv.imshow( 'a', ar )
cv.waitKey(0)
cv.destroyAllWindows()