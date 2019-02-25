from scipy.misc import face
import matplotlib.pyplot as plt

# 使用 matplotlib 显示图片
# show_image

img = face()
# img = ascent()

plt.imshow(img, extent=[-25, 25, -25, 25], cmap=plt.cm.bone)
plt.show()
