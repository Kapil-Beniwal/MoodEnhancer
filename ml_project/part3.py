import numpy as np
import cv2
from keras.preprocessing import image
import os
os.system("cls")

import tensorflow as tf
if tf.test.is_gpu_available():
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
	config.log_device_placement = True  # to log device placement (on which device the operation ran)
										# (nothing gets printed in Jupyter, only if you run it standalone)
	sess = tf.Session(config=config)
	set_session(sess)  # set this TensorFlow session as the default session for Keras

#print("\n\n#####################################################")
#print("Step-III : Inside emotion1.py --> Detecting Emotions")
#print("#####################################################\n\n")

def resource_path(relative_path):
	try:
		# PyInstaller creates a temp folder and stores path in _MEIPASS
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")
	return os.path.join(base_path, relative_path)


#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier(resource_path('haarcascade_frontalface_default.xml'))

cap = cv2.VideoCapture(0)
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open(resource_path('facial_expression_model_structure.json'), "r").read())
model.load_weights(resource_path('facial_expression_model_weights.h5')) #load weights

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
result = [0]*7
result1 = [0.0]*7

for i in range (5): 
	#ret, img = cap.read() 
	img = cv2.imread(resource_path('test\img'+str(i)+'.png'))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#print(faces) #locations of detected faces
	
	if (len(faces)==0):
		cv2.putText(img, "No face detected", (10, (1 * 20 + 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
	else:	
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])
			

			
			
			emotion = emotions[max_index]
			result[max_index] = result[max_index]+1
			result1[max_index] = result1[max_index]+round(predictions[0][max_index]*100,2)
			#print(result)
			#print(result1)
			y1=y
			y = y+20
			#write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			
			cv2.putText(img, "Emotions", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(img, "| Proability", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(img, "| Proability Bar", (200,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			
			
			bar = "....................................................."
			z = 0
			for i in range(len(predictions[0])):
				b = bar[0:int(predictions[0][i]*len(bar))]
				if (i == max_index):
					cv2.putText(img, emotions[i]+": ", (10, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
					cv2.putText(img, "| ", (100, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					cv2.putText(img, " "+str(round(predictions[0][i]*100,2)), (110, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
					cv2.putText(img, "| ", (200, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					#cv2.putText(img, " "+b, (210, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
					cv2.rectangle(img, (210, i * 20 + 10), (210 +int(predictions[0][i] * 100), (i + 1) * 20 + 4), (255, 0, 0), -1)
					
				else:	
					cv2.putText(img, emotions[i]+": ", (10, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
					cv2.putText(img, "| ", (100, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					cv2.putText(img, " "+str(round(predictions[0][i]*100,2)), (110, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
					cv2.putText(img, "| ", (200, (i + 1) * 20 + 8),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					#cv2.putText(img, " "+b, (210, (i + 1) * 20 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
					cv2.rectangle(img, (210, i * 20 + 10), (210 +int(predictions[0][i] * 100), (i + 1) * 20 + 4), (255, 0, 0), -1)

			#process on detected face end
			#-------------------------
	
	cv2.imshow('img',img)
	cv2.waitKey(0)
	"""
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	"""
#print(max(result))
if(max(result)==0):
	#print("\n\nNo real emotion in picture")
	cap.release()
	cv2.destroyAllWindows()
	
	cmd = "python part4.py "+"neutral"+""
	#print(cmd)
	os.system(cmd)
	cmd = "python guibg.py"
	os.system(cmd)
else:
	indices = [i for i, x in enumerate(result) if x == max(result)]
	#print(indices)

	max_percent = 0
	max_index =0
	if(len(indices)>1):
		for x in indices:
			if(result1[x]>max_percent):
				max_percent = result1[x]
				max_index = x
	else:
		max_index = result.index(max(result))
		max_percent = result1[max_index]

	#print(max_index," ",max_percent)
	#print(emotions[max_index])
	cap.release()
	cv2.destroyAllWindows()
	
	cmd = "python part4.py "+emotions[max_index]+""
	#print(cmd)
	os.system(cmd)
	cmd = "python guibg.py"
	os.system(cmd)
#kill open cv things		
#cap.release()
