from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from sklearn.externals import joblib
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = ["happy", "neutral","Sadness", "surprise","anger"]  # Define emotion order
clf = joblib.load('Trained_models\\AngrySad_trial.pkl')
clf2 = joblib.load('Trained_models\\HappySadSurprise_trial.pkl')

hight = 128  # height of the image
width = 64  # width of the image
hog = cv2.HOGDescriptor()
up = False
def get_face(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        print("No Face")
        return []
    images = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        images.append([img[y: y+h, x: x+w], x, y])
    images = reversed(images)
    return images

def hog_processing(face):
    face = cv2.resize(face, (width, hight), interpolation=cv2.INTER_CUBIC)  # resize images
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h = hog.compute(gray_face, winStride=(64, 128), padding=(0, 0))  # storing HOG features as column vector
    h_trans = h.transpose()  # transposing the column vector
    return h_trans[0]

def set_emojii(state, face, x, y):
    if state == 0:
        emojii = Image.open('Emojii/happy.png', 'r')
    elif state == 1:
        emojii = Image.open('Emojii/neutral.png', 'r')
    elif state == 2:
        emojii = Image.open('Emojii/sad.png', 'r')
    elif state == 3:
        emojii = Image.open('Emojii/surprise.png', 'r')
    else:
        emojii = Image.open('Emojii/angry.png', 'r')
    siz = 64, 64
    emojii.thumbnail(siz, Image.ANTIALIAS)
    background = Image.fromarray(face)
    text_img = Image.new('RGB', background.size, (0, 0, 0, 1))
    text_img.paste(background, (0, 0))
    text_img.paste(emojii, (x - 40, y - 40), mask=emojii)
    return np.array(text_img)

def quit_(root):
    root.destroy()

#def find_emotion(img):
#    faces = get_face(img)
#    if faces is not None:
#        for face in faces:
#            features = hog_processing(face[0])
#            res = np.array(features)
#            state = clf2.predict([res])
#            img = set_emojii(state, img, face[1], face[2])
#    return img

def choose_between_predicts(s1,s2):
    if s1 == 4 or s2 == 4:
        s = 4
    else:
        s = s2
    return s

def find_emotion(img):
    faces = get_face(img)
    if faces is not None:
        for face in faces:
            features = hog_processing(face[0])
            res = np.array(features)
            state1 = clf.predict([res])
            state2 = clf2.predict([res])
            state = choose_between_predicts(state1,state2)
            img = set_emojii(state, img, face[1], face[2])
    return img

def update_image(image_label, cam):
    (readsuccessful, img) = cam.read()
    if readsuccessful is False:
        return
    img = find_emotion(img)
    a = Image.fromarray(img)
    b = ImageTk.PhotoImage(image=a)
    w, h = a.size
    image_label.configure(image=b, width=w, height=h)
    image_label._image_cache = b  # avoid garbage collection
    # root.update()

def update_fps(fps_label):
    frame_times = fps_label._frame_times
    frame_times.rotate()
    frame_times[0] = time.time()
    sum_of_deltas = frame_times[0] - frame_times[-1]
    count_of_deltas = len(frame_times) - 1
    try:
        fps = int(float(count_of_deltas) / sum_of_deltas)
    except ZeroDivisionError:
        fps = 0
    fps_label.configure(text='FPS: {}'.format(fps))

def update_all(root, image_label, fps_label, cam, once):
    if once is True:
        cam = upload_video()
    update_image(image_label, cam)
    update_fps(fps_label)
    root.after(50, func=lambda: update_all(root, image_label, fps_label, cam, False))

def upload_video():
    filename = filedialog.askopenfilename(filetypes=(("MP4 files", "*.mp4;*.MP4"), ("MKV files", "*.mkv;*.MKV"), ("All files", "*.*")))
    if filename:
        try:
            return cv2.VideoCapture(filename)
        except:
            messagebox.showerror("Open Source File", "Failed to read file \n'%s'" % filename)
            return None


def upload_image(image_label):
    filename = filedialog.askopenfilename(filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")))
    if filename:
        try:
            a = Image.open(filename, 'r')
            b = ImageTk.PhotoImage(image=a)
            w, h = a.size
            image_label.configure(image=b, width=min(w, 1000), height=min(h, 500))
            image_label._image_cache = b  # avoid garbage collection
            create_process_button(np.array(a))
            root.update()
        except:
            messagebox.showerror("Open Source File", "Failed to read file \n'%s'" % filename)
            return


def create_process_button(img):
    process_button.configure(command=lambda: update_process(img))
    process_button.pack(pady=10, side="left", padx=10)

def return_all():
    image_label._image_cache = None
    image_label.configure(image=None)
    fps_label.configure(text='')
    process_button.pack_forget()
    close_button.pack_forget()
    create_webcam_button()
    create_video_button()

def create_close_image():
    close_button.pack(pady=10, side="left", padx=10)
    close_button.configure(text='Back', command=return_all)

def video_options():
    webcam_button.destroy()
    browse_button.destroy()
    video_button.destroy()
    update_all(root, image_label, fps_label, cam, True)


def webcam_options():
    video_button.destroy()
    browse_button.destroy()
    webcam_button.destroy()
    update_all(root, image_label, fps_label, cam, False)


def image_options():
    video_button.pack_forget()
    webcam_button.pack_forget()
    upload_image(image_label)
    create_close_image()


def update_process(img):
    fps_label.configure(text='Loading...')
    img = find_emotion(img)

    a = Image.fromarray(img)
    b = ImageTk.PhotoImage(image=a)
    w, h = a.size
    image_label.configure(image=b, width=w, height=h)
    image_label._image_cache = b  # avoid garbage collection
    fps_label.configure(text='Done =D')
    root.update()

def create_webcam_button():
    webcam_button.pack(pady=10, side="left", padx=10)
def create_video_button():
    video_button.pack(pady=10, side="left", padx=10)

if __name__ == '__main__':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open('Emojii/cover.png', 'r'))
    image_label = tk.Label(master=root,image = img, background='white', width=425, height=200)# label for the video frame
    image_label.pack(padx=20, pady=20)

    fps_label = tk.Label(master=root)# label for fps
    fps_label._frame_times = deque([0]*5)  # arbitrary 5 frame average FPS
    fps_label.pack()
    cam = cv2.VideoCapture(0)

    browse_button = tk.Button(master=root, text='Upload Image', command=image_options)
    browse_button.pack(side="left", padx=10, pady=10)
    webcam_button = tk.Button(master=root, text='Open Webcam', command=webcam_options)
    webcam_button.pack(pady=10, side="left", padx=10)
    video_button = tk.Button(master=root, text='Upload Video', command=video_options)
    video_button.pack(pady=10, side="left", padx=10)

    quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root))
    quit_button.pack(side="right", padx=10, pady=10)

    process_button = tk.Button(master=root, text='Find Emotions')
    close_button = tk.Button(master=root, text="Close")
    root.mainloop()