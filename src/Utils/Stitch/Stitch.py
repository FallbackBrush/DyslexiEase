import cv2
import os

image_folder = r'D:\gokul\FINAL CAPSTONE\DyslexiEase\Results\Generated Content\GAN TESTS\saved_examples\step6'
video_name = '256x256.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()