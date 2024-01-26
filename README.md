import os
import cv2
import librosa
import numpy as np
import python_speech_features

def process_video(video_path, audio_folder, image_folder, output_folder):
    # Ensure output folders exist
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # print(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Extract audio and save to audio folder
    audio_path = os.path.join(audio_folder, "audio.wav")
    print(audio_path)
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}")
    print("finish_audio")

    # Extract frames and save to image folder
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_path = os.path.join(image_folder, f"frame_{frame_count}.png")
        cv2.imwrite(image_path, frame)
        frame_count += 1

    cap.release()

    # Load audio file and extract MFCC
    speech, sr = librosa.load(audio_path, sr=16000)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)

    # Save MFCC with timestamp as filename
    for i, mfcc_frame in enumerate(mfcc):
        output_file = os.path.join(output_folder, f"frame_{i}.npy")
        np.save(output_file, mfcc_frame)

    # Clean up: remove temporary audio file
    os.remove(audio_path)

# Example usage
audio_folder = r"D:\Dataset\MEAD\M003\video (2)\save_audio"
image_folder = r"D:\Dataset\MEAD\M003\video (2)\save_image"
output_folder = r"D:\Dataset\MEAD\M003\video (2)\save_video\output_folder"



filepath = r"D:\Dataset\MEAD\M003\video (2)\video"

pathDir = os.listdir(filepath)
allp=[]
for i in range(len(pathDir)):
    emotion = pathDir[i]
    path = os.path.join(filepath,emotion)
    Dir = os.listdir(path)
    for j in range(len(Dir)):
        video_file = os.path.join(path,Dir[j], 'level_1')
        for k in range(1, 31):  # assuming you have 20 video files (adjust as needed)
            video_file = os.path.join(video_file, f"{k:03d}.mp4")

            # Do something with the constructed video_file path
            # For example, you can print it
            print(video_file)
            index = Dir[j].split('.')[0]
            # save = os.path.join(save_path,emotion+'_'+index)
            audio_folder = os.path.join(audio_folder,emotion+'_'+index)
            image_folder = os.path.join(image_folder, emotion + '_' + index)
            output_folder = os.path.join(output_folder, emotion + '_' + index)
            process_video(video_file, audio_folder, image_folder, output_folder)
            print(i, emotion, j, index)

# process_video(video_path, audio_folder, image_folder, output_folder)
