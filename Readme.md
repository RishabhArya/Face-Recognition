---


---

<h1 id="face-recogntition">Face Recogntition</h1>
<p><strong>How Face Recognition Works?</strong></p>
<p>Refer <a href="https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc1">this</a> to Know How Face Recognition works.</p>
<h2 id="face-recognition-on-static-images">Face Recognition on static Images</h2>
<p>Google Collab Link - <a href="https://colab.research.google.com/gist/RishabhArya/1b5ea79d83e0f6d9beabd5795ae9cdc7/face_recognition.ipynb#scrollTo=X8UvQd3gsUEG">Referecnce</a></p>
<h2 id="face-recognition-using-webcam">Face Recognition using WebCam</h2>
<p><strong>Project Structure</strong></p>
<pre><code>facerecognition
|
|____test images
|____facerecognize_static.py
|____facerecognize_cam.py
</code></pre>
<p><img src="https://github.com/RishabhArya/Face-Recognition/blob/master/Screenshots/Screenshot%20from%202020-06-18%2001-33-01.png" alt="screenshot1"><img src="https://github.com/RishabhArya/Face-Recognition/blob/master/Screenshots/Screenshot%20from%202020-06-18%2001-30-45.png" alt="screenshot2"></p>
<p><strong>How to Use ?</strong></p>
<pre><code>- https://github.com/RishabhArya/Face-Recognition.git
- Extract the Folder
- Open Pycharm or any other prefered IDE
- Go to File -&gt; Settings -&gt; Preferences-&gt;Project:ProjectName-&gt;
Project Interpreator -&gt; add package "face-recognition"
Authored by-Adam Geitgey
</code></pre>
<h2 id="code"><strong>Code</strong></h2>
<blockquote>
<p>facerecognize_static.py</p>
</blockquote>
<pre><code>pip install face_recognition  
  
from PIL import Image, ImageDraw  
from IPython.display import display  
  
#Training the Face Recognition Model  
import face_recognition  
import numpy as np  
from PIL import Image, ImageDraw  
from IPython.display import display  
  
  
#Load a First sample picture and learn how to recognize it.  
first_image = face_recognition.load_image_file("images/first_face_image.jpeg")  
first_face_encoding = face_recognition.face_encodings(first_image)[0]  
  
#Load a second sample picture and learn how to recognize it.  
second_image = face_recognition.load_image_file("images/second_face_image.jpeg")  
second_face_encoding = face_recognition.face_encodings(second_image)[0]  
  
#Create arrays of known face encodings and their names  
known_face_encodings = [  
    first_face_encoding,  
  second_face_encoding  
]  
known_face_names = [  
    "first person name",  
  "second person name"  
]  
  
#Detect the face recognition model  
  
#Load an image with an unknown face  
unknown_image = face_recognition.load_image_file("image_to_recognize.jpeg")  
  
#Find all the faces and face encodings in the unknown image  
face_locations = face_recognition.face_locations(unknown_image)  
print(face_locations)  
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)  
  
#Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library  
#See http://pillow.readthedocs.io/ for more about PIL/Pillow  
pil_image = Image.fromarray(unknown_image)  
#Create a Pillow ImageDraw Draw instance to draw with  
draw = ImageDraw.Draw(pil_image)  
  
#Loop through each face found in the unknown image  
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):  
    #See if the face is a match for the known face(s)  
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  
  
    name = "Unknown"  
  
  #Or instead, use the known face with the smallest distance to the new face  
  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)  
    best_match_index = np.argmin(face_distances)  
    if matches[best_match_index]:  
        name = known_face_names[best_match_index]  
  
    #Draw a box around the face using the Pillow module  
  draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))  
  
    #Draw a label with a name below the face  
  text_width, text_height = draw.textsize(name)  
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))  
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))  
  
  ![enter image description here](https://github.com/RishabhArya/Face-Recognition/blob/master/Screenshots/Screenshot%20from%202020-06-19%2007-36-01.png)
  
#Remove the drawing library from memory as per the Pillow docs  
del draw  
  
#Display the resulting image  
display(pil_image)
</code></pre>
<blockquote>
<p>facerecognize_cam.py</p>
</blockquote>
<pre><code>import face_recognition  
import cv2  
import numpy as np  
  
#This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the #other example, but it includes some basic performance tweaks to make things run a lot faster: #1. Process each video frame at 1/4 resolution (though still display it at full resolution) #2. Only detect faces in every other frame of video.   
#PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam. #OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this #specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.   
#Get a reference to webcam #0 (the default one) video_capture = cv2.VideoCapture(0)  
  
#Load a sample picture and learn how to recognize it. first_image = face_recognition.load_image_file("images/first_person_image.jpg")  
first_face_encoding = face_recognition.face_encodings(first_image)[0]  
  
#Load a second sample picture and learn how to recognize it. second_image = face_recognition.load_image_file("images/second_person_image.jpg")  
second_face_encoding = face_recognition.face_encodings(second_image)[0]  
  
#Create arrays of known face encodings and their names known_face_encodings = [  
    first_face_encoding,  
  second_face_encoding  
]  
known_face_names = [  
    "first person name",  
  "second person name"  
]  
  
#Initialize some variables face_locations = []  
face_encodings = []  
face_names = []  
process_this_frame = True  
  
while True:  
    #Grab a single frame of video    
 ret, frame = video_capture.read()  
  
    #Resize frame of video to 1/4 size for faster face recognition processing    
 small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  
  
    #Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)    
 rgb_small_frame = small_frame[:, :, ::-1]  
  
    #Only process every other frame of video to save time    
 if process_this_frame:  
    #Find all the faces and face encodings in the current frame of video    
 face_locations = face_recognition.face_locations(rgb_small_frame)  
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  
  
    face_names = []  
    for face_encoding in face_encodings:  
#See if the face is a match for the known face(s) matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  
name = "Unknown"  
  
##If a match was found in known_face_encodings, just use the first one. #if True in matches: #first_match_index = matches.index(True) #name = known_face_names[first_match_index] #Or instead, use the known face with the smallest distance to the new face  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding) best_match_index = np.argmin(face_distances)  
if matches[best_match_index]:  
    name = known_face_names[best_match_index]  
  
face_names.append(name)  
  
process_this_frame = not process_this_frame  
  
#Display the results for (top, right, bottom, left), name in zip(face_locations, face_names):  
#Scale back up face locations since the frame we detected in was scaled to 1/4 size top *= 4  
right *= 4  
bottom *= 4  
left *= 4  
  
#Draw a box around the face cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
  
#Draw a label with a name below the face cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  
font = cv2.FONT_HERSHEY_DUPLEX  
cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  
  
#Display the resulting image cv2.imshow('Video', frame)  
  
#Hit 'q' on the keyboard to quit! if cv2.waitKey(1) &amp; 0xFF == ord('q'):  
    break  
  
  #Release handle to the webcam video_capture.release()  
cv2.destroyAllWindows()
</code></pre>
<p><img src="https://github.com/RishabhArya/Face-Recognition/blob/master/Screenshots/Screenshot%20from%202020-06-04%2008-20-39.png" alt="output"></p>
<h2 id="more-examples">More Examples</h2>
<p>All the examples are available <a href="https://github.com/ageitgey/face_recognition/tree/master/examples">here</a>.</p>
<h4 id="face-detection">Face Detection</h4>
<ul>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py">Find faces in a photograph</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py">Find faces in a photograph (using deep learning)</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py">Find faces in batches of images w/ GPU (using deep learning)</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py">Blur all the faces in a live video using your webcam (Requires OpenCV to be installed)</a></li>
</ul>
<h4 id="facial-features">Facial Features</h4>
<ul>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py">Identify specific facial features in a photograph</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py">Apply (horribly ugly) digital make-up</a></li>
</ul>
<h4 id="facial-recognition">Facial Recognition</h4>
<ul>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py">Find and recognize unknown faces in a photograph based on photographs of known people</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py">Identify and draw boxes around each person in a photo</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py">Compare faces by numeric face distance instead of only True/False matches</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py">Recognize faces in live video using your webcam - Simple / Slower Version (Requires OpenCV to be installed)</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py">Recognize faces in live video using your webcam - Faster Version (Requires OpenCV to be installed)</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py">Recognize faces in a video file and write out new video file (Requires OpenCV to be installed)</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py">Recognize faces on a Raspberry Pi w/ camera</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py">Run a web service to recognize faces via HTTP (Requires Flask to be installed)</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py">Recognize faces with a K-nearest neighbors classifier</a></li>
<li><a href="https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py">Train multiple images per person then recognize faces using a SVM</a></li>
</ul>
<h2 id="reference-section">Reference Section</h2>
<ul>
<li>
<p><a href="https://github.com/RishabhArya/Face-Recognition/blob/master/Readme.md">My GitHub Repo</a></p>
</li>
<li>
<p><a href="https://github.com/ageitgey/face_recognition">Official Face Recognition Repo module</a></p>
</li>
<li>
<p><a href="https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78">Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning</a></p>
</li>
<li>
<p><a href="https://medium.com/analytics-vidhya/a-take-on-h-o-g-feature-descriptor-e839ebba1e52">A Take on H.O.G Feature Descriptor - Analytics Vidhya - Medium</a></p>
</li>
<li>
<p><a href="https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html">OpenCV: Face Detection using Haar Cascades</a></p>
</li>
<li>
<p>My article on how Face Recognition works: <a href="https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78">Modern Face Recognition with Deep Learning</a></p>
<ul>
<li>Covers the algorithms and how they generally work</li>
</ul>
</li>
<li>
<p><a href="https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/">Face recognition with OpenCV, Python, and deep learning</a> by Adrian Rosebrock</p>
<ul>
<li>Covers how to use face recognition in practice</li>
</ul>
</li>
<li>
<p><a href="https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/">Raspberry Pi Face Recognition</a> by Adrian Rosebrock</p>
<ul>
<li>Covers how to use this on a Raspberry Pi</li>
</ul>
</li>
<li>
<p><a href="https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/">Face clustering with Python</a> by Adrian Rosebrock</p>
<ul>
<li>Covers how to automatically cluster photos based on who appears in each photo using unsupervised learning</li>
</ul>
</li>
</ul>

