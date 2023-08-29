import cv2
import os

# Function to extract frames
def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    filename = path.split('\\')[-1]
    basepath = r"D:\gokul\FINAL CAPSTONE\DyslexiEase\Data\Processsed"
    if not os.path.exists(os.path.join(basepath,filename)):
        os.mkdir(os.path.join(basepath,filename))
    while success:
        success, image = vidObj.read()
        #cv2.imwrite("frame%d.jpg" % count, image)
        cv2.imwrite(os.path.join(basepath,filename,"frame%d.jpg" % count),image)
        count += 1
    print("done with video")

if __name__ == '__main__':
    
	FrameCapture(r"D:\gokul\FINAL CAPSTONE\DyslexiEase\src\Utils\frameExtract\Videos\test.mp4")
