import random
from PIL import Image
import numpy as np
import face_recognition

class FaceTracker():
    def __init__(self, options):
        self.known_faces = {}
        self.index = 0
        pass

    # Process an input_image
    def process(self, input_image):
        input_image = np.array(input_image)
        height, width = input_image.shape[0:2]
        
        face_locations = face_recognition.face_locations(input_image)
        face_encodings = face_recognition.face_encodings(input_image, face_locations)
        
        found_faces = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([self.known_faces[f] for f in self.known_faces], face_encoding)
            
            # if there is a match, return the matched index
            if True in matches:
                face_index = matches.index(True)
                
            
            # if there is no match, save the face encoding to known faces dict
            else:
                self.known_faces[self.index] = face_encoding
                face_index = self.index
                self.index += 1
            
            top, right, bottom, left = face_location
            box = [ float(left) / width, float(top) / height, float(right) / width, float(bottom) / height]
            
            found_faces.append({
                "index": face_index,
                "box": box
            })
        return found_faces