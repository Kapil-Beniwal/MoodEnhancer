import os 
#os.system('&> /dev/null')
os.system("cls")
#print("\n\n###############################################")
#print("Step-II : Inside part1.py --> Clicking Photos")
#print("##############################################\n\n")


def resource_path(relative_path):
	try:
		# PyInstaller creates a temp folder and stores path in _MEIPASS
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")
	return os.path.join(base_path, relative_path)


import cv2
from time import sleep
cap = cv2.VideoCapture(0)
for i in range (5):
	while(cap.isOpened()):
		ret,img1 = cap.read()
		#cv2.imshow('cap',img)
		cv2.imwrite(resource_path('test\img'+str(i)+'.png'),img1)
		#print('\n\nimg'+str(i+1)+'.png'+" is saved")
		break
	sleep(1)
cap.release()
cv2.destroyAllWindows()
os.system("cls")
#print("part2 complete")

#cmd = "py part3.py"
#os.system(cmd)


