import pytesseract
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\andriimatsevytyi\\AppData\\Local\\Programs\\Tesseract\\tesseract.exe'
nltk.download('punkt')

# video input (webcam, can be changed to separate detached camera)
video_input = cv2.VideoCapture(0)

# params for optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ShiTomasi corner detection params to detect key points
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

old_gray = None
p0 = None

# preprocess frame for better text extraction
def frame_preprocessor(frame):
    frame = cv2.resize(frame, (800, 800))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey

# Segment frame into blocks of text
def frame_text_blocks(frame):
    _, thresh1 = cv2.threshold(frame, 140, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # display contours
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Contour Image")

    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Filtering out error contours
            valid_contours.append(cnt)
            plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()
    return valid_contours, hierarchy

# apply OCR on segmented text blocks
def extract_text(frame, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = frame[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        print(f"Extracted text: {text}")

# process a single text block and save results
def process_frame(frame, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = frame[y:y + h, x:x + w]
    text = pytesseract.image_to_string(cropped, config='--psm 7')

    with open("test_recognized.txt", "a") as file:
        file.write(f"{text}\n")

# optical flow implementation using Lucas-Kanade method
def apply_optical_flow(old_frame, old_gray, new_frame):
    global p0 
    
    # detect initial points if needed
    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)
    
    # calculation itself
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
        
        # successful points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        mask = np.zeros_like(new_frame)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            new_frame = cv2.circle(new_frame, (a, b), 5, (0, 0, 255), -1)

        p0 = good_new.reshape(-1, 1, 2)
    
    return new_frame, new_gray

# contextual text grouping

def contextual_grouping(found_text):
    grouped_texts = []
    visited = set()  # To avoid processing the same text again

    # Extract the individual texts for vectorization
    texts = [text for text, _ in found_text]
    
    # Vectorize the texts using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    
    for i, (text, (x, y, w, h)) in enumerate(found_text):
        if i in visited:
            continue

        # Start a new group
        group = [text]
        group_rect = (x, y, w, h)  # Initialize bounding box for grouping
        
        for j, (other_text, (ox, oy, ow, oh)) in enumerate(found_text):
            if i != j and j not in visited:
                # Check if the contours are close enough to be grouped
                if (abs(x - ox) < 50 and abs(y - oy) < 50):  # Example threshold of 50 pixels
                    # Calculate cosine similarity
                    similarity = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                    if similarity > 0.5:  # Example threshold for similarity
                        group.append(other_text)
                        group_rect = (min(group_rect[0], ox), min(group_rect[1], oy),
                                      max(group_rect[2], ox + ow), max(group_rect[3], oy + oh))
                        visited.add(j)

        # Combine the group texts into one
        combined_text = "\n".join(group)
        grouped_texts.append((combined_text, group_rect))

    return grouped_texts

if __name__ == "__main__":
    ret, old_frame = video_input.read()
    old_gray = frame_preprocessor(old_frame)  # Initialize first frame
    
    while True:
        ret, frame = video_input.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        """
        
        BASIC PREPROCESSING AND OPTICAL FLOW
        
        """
        
        frame, old_gray = apply_optical_flow(old_frame, old_gray, frame)
        
        processed_frame = frame_preprocessor(frame)
        
        # Detect text blocks
        contours, hierarchy = frame_text_blocks(processed_frame)
        
        """
        
        NLP FOR CONTEXTUAL GROUPING
        
        """
        
        found_text = extract_text(processed_frame, contours)
        
        grouped_texts = contextual_grouping(found_text)
        
        # internal call to the glove
        # Glove.execute(grouped_texts[0])
        print(grouped_texts[0])
        
        # frame with optical flow visualization
        cv2.imshow('Video Input with Optical Flow', frame)
        
        old_frame = frame
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_input.release()
    cv2.destroyAllWindows()

