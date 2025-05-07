# import os

# UPLOAD_FOLDER = "upload_folder"

# def count_images():
#     num_images = len([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".jpg", ".jpeg", ".png",".pdf"))])
#     print(f"Number of images in the folder: {num_images}")
    

# if __name__ == "__main__":
#     count_images()


# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plts
# import gc




# def stitch_images(img1, img2, technique):
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     # Convert to grayscale
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     if technique.lower()=="sift":
#         #Use SIFT for feature detection
#         sift = cv2.SIFT_create()
#         kp1, des1 = sift.detectAndCompute(img1, None)
#         kp2, des2 = sift.detectAndCompute(img2, None)

#         #Use FLANN-based matcher
#         index_params = dict(algorithm=1, trees=5)
#         search_params = dict(checks=50)
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
    
#         matches = flann.knnMatch(des1, des2, k=2)

#     elif technique.lower()=="orb":
#         # Initialize ORB detector
#         orb = cv2.ORB_create(nfeatures=5000)  # Increase nfeatures for more keypoints
        
#         # Detect and compute features
#         kp1, des1 = orb.detectAndCompute(gray1, None)
#         kp2, des2 = orb.detectAndCompute(gray2, None)
        
        
#         # # Initialize Brute-Force Matcher
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
#         # Match descriptors
#         matches = bf.knnMatch(des1, des2, k=2)  # k=2 for Lowe's Ratio Test
            
        
#     # Apply Lowe's ratio test
#     good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
#     print("Good matches:", len(good_matches))
#     # Skip if there aren't enough good matches
#     if len(good_matches) < 25:
#         print("Insufficient matches. Skipping this pair.")
#         return None
    
#     # Extract points
#     pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     # Find homography matrix
#     H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)


#     print(H)
#     # Warp the first image
#     height1, width1 = img1.shape[:2]
#     height2, width2 = img2.shape[:2]

#     # Calculate the size of the output image
#     corners1 = np.float32([[0, 0], [width1, 0], [width1, height1], [0, height1]]).reshape(-1, 1, 2)
#     warped_corners1 = cv2.perspectiveTransform(corners1, H)
#     corners2 = np.float32([[0, 0], [width2, 0], [width2, height2], [0, height2]]).reshape(-1, 1, 2)
#     all_corners = np.concatenate((warped_corners1, corners2), axis=0)
#     # # print("corners 1:", corners1)
#     # # print()
#     # # print("warpedcorners 1:", warped_corners1)
#     # # print()
#     # # print("corners 2:", corners2)
#     # # print()
#     # # print("corners 1:", corners1)
#     # # print()
#     matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
#     plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
#     plt.show()
    
#     [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
#     [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

#     # Translation matrix
#     translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

#     # Warp the first image with translation applied
#     warped_img1 = cv2.warpPerspective(img1, translation @ H, (x_max - x_min, y_max - y_min))

#     # Place the second image in the same canvas
#     canvas = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
#     canvas[-y_min:height2 - y_min, -x_min:width2 - x_min] = img2

#     # Combine both images
#     stitched_image = np.maximum(warped_img1, canvas)

#     del img1, img2, kp1, kp2, des1, des2, warped_corners1, warped_img1, translation, matched_img
#     gc.collect()
#     return stitched_image


# def resize_images(images, scale_percent=30):
#     resized = []
#     for img_path in images:
#         img = cv2.imread(img_path)
#         width = int(img.shape[1] * scale_percent / 100)
#         height = int(img.shape[0] * scale_percent / 100)
#         resized.append(cv2.resize(img, (width, height)))
#     return resized

# def crop_image(image, top_margin, bottom_margin, left_margin, right_margin):
#         """
#         Crop specified margins from each side of the image.
#         """
#         height, width = image.shape[:2]
#         return image[
#             top_margin:height - bottom_margin,
#             left_margin:width - right_margin
#         ]


# def stitch_folder_images(folder_path):
#     # Load images
#     image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
#     image_files.sort()  # Sort files for consistency
#     #image_files= sort_images_by_gps(folder_path)
#     print("Images", image_files)
#     if len(image_files) < 2:
#         print("Need at least two images to perform stitching.")
#         return

#     # Resize images
#     images = resize_images(image_files, scale_percent=30)
    
#     print("cropping image borders")
#     for i in range(len(images)):
#         images[i]= crop_image(images[i],100,100,100,100)

#     print("strating stitch")
#     # Iteratively stitch images
#     intermediate_results = []
#     for i in range(0, len(images) - 1,2):  # Process in pairs  #add step 2 originally
#         print(f"Stitching image {i} and image {i+1}")
#         stitched = stitch_images(images[i], images[i + 1],"sift")
#         if stitched is None:  # Retry with ORB
#             print(f"SIFT failed, attempting ORB for image {i} and image {i+1}")
#             stitched = stitch_images(images[i], images[i + 1], "orb")
#             if stitched == None:
#                 intermediate_results.append(images[i])
#                 intermediate_results.append(images[i+1])
#         # if stitched is None:  # Retry with LoFTR
#         #     print(f"Both SIFT and ORB failed. Trying LoFTR.")
#         #     stitched = stitch_images(images[i], images[i + 1],"loftr")
                
        
#         if stitched is not None:
#             stitched=crop_black_borders(stitched)
#             plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
#             plt.axis('off')
#             plt.show()
#             intermediate_results.append(stitched)
        
        
#     print("start intermediate")
#     # Handle remaining image if the count is odd
#     if len(images) % 2 != 0:
#         intermediate_results.append(images[-1])

#     # Stitch intermediate results
#     while len(intermediate_results) > 1:
#         next_round = []
#         for i in range(0, len(intermediate_results) - 1,2 ): #add step 2 originally
#             print(f"Stitching intermediate {i} and {i+1}")
#             stitched = stitch_images(intermediate_results[i], intermediate_results[i + 1],"sift")
#             if stitched is None:
#                     intermediate_results.append(images[i])
#                     intermediate_results.append(images[i+1])
                
#             if stitched is not None:
#                 # plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
#                 # plt.axis('off')
#                 # plt.show()
#                 next_round.append(stitched)
#         if len(intermediate_results) % 2 != 0:
#             next_round.append(intermediate_results[-1])
#         intermediate_results = next_round

#     # Final stitched image
#     final_image = intermediate_results[0]
#     cv2.imwrite("final_stitched_output.jpg", final_image)
#     plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
#     plt.title("Final Output")
#     plt.axis('off')
#     plt.show()
#     print("Stitching complete.")


# # Path to the folder containing images


# if __name__ == "__main__":
#     folder_path = './upload_folder'  # Update with your folder path
#     stitch_folder_images(folder_path)


# # import os
# # import time  # Import the time module for adding delays

# # UPLOAD_FOLDER = "upload_folder"

# # def count_images():
# #     # Wait for 40 seconds
# #     print("Waiting for 40 seconds...")
# #     time.sleep(40)  # Pause the program for 40 seconds
    
# #     # Count the images
# #     num_images = len([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".jpg", ".jpeg", ".png", ".pdf"))])
# #     print(f"Number of images in the folder: {num_images}")

# # if __name__ == "__main__":
# #     count_images()








# import cv2
# import numpy as np
# import os

# def stitch_images(images):
#     # Create a stitcher object
#     stitcher = cv2.Stitcher_create()

#     # Perform stitching
#     status, stitched_image = stitcher.stitch(images)

#     if status == cv2.Stitcher_OK:
#         print("Stitching successful!")
#         return stitched_image
#     else:
#         print("Stitching failed with error code:", status)
#         return None

# def load_images_from_folder(folder_path):
#     # List all image files in the folder
#     image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

#     # Sort the files (optional, to ensure order for stitching)
#     image_files.sort()

#     # Read the images
#     images = []
#     for image_file in image_files:
#         image_path = os.path.join(folder_path, image_file)
#         img = cv2.imread(image_path)
#         if img is not None:
#             images.append(img)

#     return images

# if __name__ == "__main__":
#     # Set the folder path containing your images
#     folder_path = './upload_folder'  # Replace with your folder path

#     # Load images from the folder
#     images = load_images_from_folder(folder_path)

#     if len(images) < 2:
#         print("Not enough images to stitch. Please add at least two images.")
#     else:
#         # Call the stitch function
#         stitched_result = stitch_images(images)

#         if stitched_result is not None:
#             # Save or display the stitched image
#             output_path = 'stitched_output.jpg'
#             cv2.imwrite(output_path, stitched_result)
#             print(f"Stitched image saved as {output_path}")
#             cv2.imshow('Stitched Image', stitched_result)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()






import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import time
from flask import jsonify,session



class Counter:
    file_path = "counter.txt"  # File to store counter value

    @staticmethod
    def _read_counter():
        """Reads the counter value from the file."""
        try:
            with open(Counter.file_path, "r") as file:
                return int(file.read().strip())
        except (FileNotFoundError, ValueError):
            return 0  # Default to 0 if file doesn't exist or has invalid data

    @staticmethod
    def _write_counter(value):
        """Writes the counter value to the file."""
        with open(Counter.file_path, "w") as file:
            file.write(str(value))

    @staticmethod
    def increment():
        """Increments and saves the counter value."""
        value = Counter._read_counter() + 1
        Counter._write_counter(value)
        return value

# Usage Example
counter = Counter()
  # This will retain value even after restarting the program


counter = Counter()
def stitch_images(img1, img2, technique):
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Convert to grayscale
    print("Statjkjfkdasjf")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if technique.lower()=="sift":
        #Use SIFT for feature detection
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        #Use FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
    
        matches = flann.knnMatch(des1, des2, k=2)

    elif technique.lower()=="orb":
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=5000)  # Increase nfeatures for more keypoints
        
        # Detect and compute features
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        
        # # Initialize Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)  # k=2 for Lowe's Ratio Test
            
        
    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        # print("Good matches:", len(good_matches))
    # Skip if there aren't enough good matches
    if len(good_matches) < 25:
        # print("Insufficient matches. Skipping this pair.")
        return None
    
    # Extract points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)


    # print(H)
    # Warp the first image
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Calculate the size of the output image
    corners1 = np.float32([[0, 0], [width1, 0], [width1, height1], [0, height1]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, H)
    corners2 = np.float32([[0, 0], [width2, 0], [width2, height2], [0, height2]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    # # print("corners 1:", corners1)
    # # print()
    # # print("warpedcorners 1:", warped_corners1)
    # # print()
    # # print("corners 2:", corners2)
    # # print()
    # # print("corners 1:", corners1)
    # # print()
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    # plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp the first image with translation applied
    warped_img1 = cv2.warpPerspective(img1, translation @ H, (x_max - x_min, y_max - y_min))

    # Place the second image in the same canvas
    canvas = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    canvas[-y_min:height2 - y_min, -x_min:width2 - x_min] = img2

    # Combine both images
    stitched_image = np.maximum(warped_img1, canvas)

    del img1, img2, kp1, kp2, des1, des2, warped_corners1, warped_img1, translation, matched_img
    gc.collect()
    return stitched_image


def resize_images(images, scale_percent=30):
    resized = []
    for img_path in images:
        img = cv2.imread(img_path)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        resized.append(cv2.resize(img, (width, height)))
    return resized

def crop_image(image, top_margin, bottom_margin, left_margin, right_margin):
        """
        Crop specified margins from each side of the image.
        """
        height, width = image.shape[:2]
        return image[
            top_margin:height - bottom_margin,
            left_margin:width - right_margin
        ]


def stitch_folder_images(folder_path):
    # print(f"Received parameter: {sys.argv[1]}")

    # Load images
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    image_files.sort()  # Sort files for consistency
    #image_files= sort_images_by_gps(folder_path)
    # print("Images", image_files)
    if len(image_files) < 2:
        # print("Need at least two images to perform stitching.")
        return

    # Resize images
    images = resize_images(image_files, scale_percent=30)
    


    # print("strating stitch")
    # Iteratively stitch images
    intermediate_results = []
    for i in range(0, len(images) - 1,2):  # Process in pairs  #add step 2 originally
        # print(f"Stitching image {i} and image {i+1}")
        stitched = stitch_images(images[i], images[i + 1],"sift")
        if stitched is None:  # Retry with ORB
            # print(f"SIFT failed, attempting ORB for image {i} and image {i+1}")
            # stitched = stitch_images(images[i], images[i + 1], "orb")
            # if stitched is None:
                intermediate_results.append(images[i])
                intermediate_results.append(images[i+1])
        # if stitched is None:  # Retry with LoFTR
        #     print(f"Both SIFT and ORB failed. Trying LoFTR.")
        #     stitched = stitch_images(images[i], images[i + 1],"loftr")
                
        
        if stitched is not None:
            # stitched=crop_black_borders(stitched)
            # plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()
            intermediate_results.append(stitched)
        
        
    # print("start intermediate")
    # Handle remaining image if the count is odd
    if len(images) % 2 != 0:
        intermediate_results.append(images[-1])

    # Stitch intermediate results
    while len(intermediate_results) > 1:
        next_round = []
        for i in range(0, len(intermediate_results) - 1,2 ): #add step 2 originally
            # print(f"Stitching intermediate {i} and {i+1}")
            stitched = stitch_images(intermediate_results[i], intermediate_results[i + 1],"sift")
            if stitched is None:
                    next_round.append(intermediate_results[i])
                    next_round.append(intermediate_results[i+1])
                    # continue
                
            if stitched is not None:
                # stitched=crop_black_borders(stitched)
                # plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.show()
                next_round.append(stitched)
        
        if len(intermediate_results) % 2 != 0:
            next_round.append(intermediate_results[-1])
        intermediate_results = next_round

    # Final stitched image
    final_image = intermediate_results[0]
    local =counter.increment()
    print(f"static/final_stitched_output_{local}_{sys.argv[1]}.jpg")
    cv2.imwrite(f"static/final_stitched_output_{local}_{sys.argv[1]}.jpg", final_image)
    # plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    # plt.title("before crop Final Output")
    # plt.axis('off')
    # plt.show()

    # final_image = crop_black_borders(final_image)
    # cv2.imwrite("final_stitched_output_cropped.jpg", final_image)
    # plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    # plt.title("After Crop Final Output")
    # plt.axis('off')
    # plt.show()

# Path to the folder containing images
folder_path = './uploads'  # Update with your folder path


start = time.time()
stitch_folder_images(folder_path)
end = time.time()
total = end - start
# print("Time taken to stitch images:", total)
# print()