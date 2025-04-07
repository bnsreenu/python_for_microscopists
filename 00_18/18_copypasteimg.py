
from PIL import Image
img00 = Image.open(r"python_for_microscopists\images\test_image.jpg")
img00.size
img02 = Image.open(r"python_for_microscopists\images\leaf.jpg")
#img02.size
img02.thumbnail((100,100))
img02.show()
img00.paste(img02, (13,13))
img00.show()
