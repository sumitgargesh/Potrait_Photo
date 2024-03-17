import cv2
import os
import shutil

def detect_faces(image_path, min_face_width=500):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Filter out faces larger than min_face_width
    large_faces = [face for face in faces if face[2] > min_face_width]

    return large_faces

def process_images_in_directory(directory_path):
    # List all files in the directory
    image_files = [file for file in os.listdir(directory_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)

        # Detect large faces in the image
        large_faces = detect_faces(image_path)

        # Print information about the detected faces
        if large_faces:
            # Create the full file paths
            source_copy_dir = directory_path
            destination_copy_dir = "D:/Ideas/ML model to convert simple image into potrait image/helen dataset/Large Faces Folder"
            source_copy_path = os.path.join(source_copy_dir, image_file)
            destination_copy_path = os.path.join(destination_copy_dir, image_file)
            # print(f"Large faces found in {image_file}: {large_faces}")

            # Copy the file
            shutil.copy2(source_copy_path, destination_copy_path)
            print(f"File '{image_file}' copied to '{destination_copy_dir}'.")
        else:
            print(f"No large faces found in {image_file}")

# Specify the directory containing your images
image_directory = "D:/Ideas/ML model to convert simple image into potrait image/helen dataset/helen_1"

# Process images in the specified directory
process_images_in_directory(image_directory)
