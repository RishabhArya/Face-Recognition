# Face Recogntition
**How Face Recognition Works?**
Refer [this](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc1) to Know How Face Recognition works.
## Face Recognition on static Images
Google Collab Link - [Referecnce](https://colab.research.google.com/gist/RishabhArya/1b5ea79d83e0f6d9beabd5795ae9cdc7/face_recognition.ipynb#scrollTo=X8UvQd3gsUEG)

## Face Recognition using WebCam

**Project Structure**

    facerecognition
    |
    |____test images
    |____facerecognize_static.py
    |____facerecognize_cam.py

![screenshot1](https://github.com/RishabhArya/Face-Recognition/blob/master/Screenshots/Screenshot%20from%202020-06-18%2001-33-01.png)![screenshot2](https://github.com/RishabhArya/Face-Recognition/blob/master/Screenshots/Screenshot%20from%202020-06-18%2001-30-45.png)

**How to Use ?**

    - https://github.com/RishabhArya/Face-Recognition.git
    - Extract the Folder
    - Open Pycharm or any other prefered IDE
    - Go to File -> Settings -> Preferences->Project:ProjectName->
    Project Interpreator -> add package "face-recognition"
    Authored by-Adam Geitgey
## **Code**

> facerecognize_static.py


    pip install face_recognition
    
    from PIL import Image, ImageDraw
    from IPython.display import display
    
    # Training the Face Recognition Model
    import face_recognition
    import numpy as np
    from PIL import Image, ImageDraw
    from IPython.display import display
    
    
    # Load a First sample picture and learn how to recognize it.
    first_image = face_recognition.load_image_file("images/first_face_image.jpeg")
    first_face_encoding = face_recognition.face_encodings(first_image)[0]
    
    # Load a second sample picture and learn how to recognize it.
    second_image = face_recognition.load_image_file("images/second_face_image.jpeg")
    second_face_encoding = face_recognition.face_encodings(second_image)[0]
    
    # Create arrays of known face encodings and their names
    known_face_encodings = [
        first_face_encoding,
        second_face_encoding
    ]
    known_face_names = [
        "first person name",
        "second person name"
    ]
    
    # Detect the face recognition model
    
    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file("image_to_recognize.jpeg")
    
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    print(face_locations)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
        name = "Unknown"
    
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
    
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    
    
    # Remove the drawing library from memory as per the Pillow docs
    del draw
    
    # Display the resulting image
    display(pil_image)

> facerecognize_cam.py

    import face_recognition  
    import cv2  
    import numpy as np  
      
    # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the  
    # other example, but it includes some basic performance tweaks to make things run a lot faster:  
    #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)  
    #   2. Only detect faces in every other frame of video.  
      
    # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.  
    # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this  
    # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.  
      
    # Get a reference to webcam #0 (the default one)  
    video_capture = cv2.VideoCapture(0)  
      
    # Load a sample picture and learn how to recognize it.  
    first_image = face_recognition.load_image_file("images/first_person_image.jpg")  
    first_face_encoding = face_recognition.face_encodings(first_image)[0]  
      
    # Load a second sample picture and learn how to recognize it.  
    second_image = face_recognition.load_image_file("images/second_person_image.jpg")  
    second_face_encoding = face_recognition.face_encodings(second_image)[0]  
      
    # Create arrays of known face encodings and their names  
    known_face_encodings = [  
        first_face_encoding,  
      second_face_encoding  
    ]  
    known_face_names = [  
        "first person name",  
      "second person name"  
    ]  
      
    # Initialize some variables  
    face_locations = []  
    face_encodings = []  
    face_names = []  
    process_this_frame = True  
      
    while True:  
        # Grab a single frame of video  
      ret, frame = video_capture.read()  
      
        # Resize frame of video to 1/4 size for faster face recognition processing  
      small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  
      
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)  
      rgb_small_frame = small_frame[:, :, ::-1]  
      
        # Only process every other frame of video to save time  
      if process_this_frame:  
            # Find all the faces and face encodings in the current frame of video  
      face_locations = face_recognition.face_locations(rgb_small_frame)  
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  
      
            face_names = []  
            for face_encoding in face_encodings:  
                # See if the face is a match for the known face(s)  
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  
                name = "Unknown"  
      
      # # If a match was found in known_face_encodings, just use the first one.  
     # if True in matches: #     first_match_index = matches.index(True) #     name = known_face_names[first_match_index]  
     # Or instead, use the known face with the smallest distance to the new face  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)  
                best_match_index = np.argmin(face_distances)  
                if matches[best_match_index]:  
                    name = known_face_names[best_match_index]  
      
                face_names.append(name)  
      
        process_this_frame = not process_this_frame  
      
      
        # Display the results  
      for (top, right, bottom, left), name in zip(face_locations, face_names):  
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size  
      top *= 4  
      right *= 4  
      bottom *= 4  
      left *= 4  
      
      # Draw a box around the face  
      cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
      
            # Draw a label with a name below the face  
      cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  
            font = cv2.FONT_HERSHEY_DUPLEX  
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  
      
        # Display the resulting image  
      cv2.imshow('Video', frame)  
      
        # Hit 'q' on the keyboard to quit!  
      if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
      
    # Release handle to the webcam  
    video_capture.release()  
    cv2.destroyAllWindows()

