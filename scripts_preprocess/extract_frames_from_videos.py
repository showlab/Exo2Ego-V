import json
import cv2
import os
import shutil

def find_video_directory(root_directory, target_filename):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(target_filename):
                return file

    return None

def extract_frames(video_path, output_folder, start_time, end_time, fps=30):

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, fps)
    start_frame = int(start_time * fps)
    if end_time!=-1:
        end_frame = int(end_time * fps)
    else:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame%4==0:
            output_file_path = f"{output_folder}/frame_{current_frame}.jpg"
            print(current_frame,output_file_path)
            original_height, original_width = frame.shape[:2]
            if original_height == original_width:
                target_height, target_width = 256, 256
            else:
                target_height, target_width = 270, 480
            resized_image = cv2.resize(frame, (target_width, target_height))
            cv2.imwrite(output_file_path, resized_image)
        
        current_frame += 1
    cap.release()

json_file_path_train = '"Your Data Folder"/annotations/keystep_train.json'
json_file_path_val = '"Your Data Folder"/annotations/keystep_val.json'
final_path = "Your Output Folder"
original_path = '"Your Data Folder"/takes'
scene = "category" # Please choose a category.
json_path = '"Your Data Folder"/takes.json'

with open(json_file_path_train, 'r') as file:
    json_data = json.load(file)

names = []

with open(json_path, 'r') as file:
    datas = json.load(file)
    for data in datas:
        if scene in data['take_name']:
            names.append(data['take_name'])
segments={}
names_action = []
for key,value in json_data['annotations'].items():
    if value['take_name'] in names:
        segments[value['take_name']] = value['segments']
        if value['take_name'] not in names_action:
            names_action.append(value['take_name'])

with open(json_file_path_val, 'r') as file:
    json_data = json.load(file)

for key,value in json_data['annotations'].items():
    if value['take_name'] in names:
        segments[value['take_name']] = value['segments']
        if value['take_name'] not in names_action:
            names_action.append(value['take_name'])

segments = dict(sorted(segments.items()))

names_noaction = []
for name in names:
    if name not in names_action:
        names_noaction.append(name)

for key,datas in segments.items():
    print(key)
    for i,data in enumerate(datas):
        video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/'.format(key))
        target_filename = "214-1.mp4" 
        ss = find_video_directory(video_path,target_filename)
        if ss is None:
            break
        video_path = os.path.join(video_path,ss)
        output_folder =os.path.join(final_path,'{}/ego'.format(key),str(i))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            start_time = data['start_time']
            end_time = data['end_time']
            extract_frames(video_path, output_folder, start_time, end_time)

        video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam01.mp4'.format(key))
        output_folder =os.path.join(final_path,'{}/exo'.format(key),str(i))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            start_time = data['start_time']
            end_time = data['end_time']
            extract_frames(video_path, output_folder, start_time, end_time)

        video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam02.mp4'.format(key))
        output_folder =os.path.join(final_path,'{}/exo2'.format(key),str(i))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            start_time = data['start_time']
            end_time = data['end_time']
            if os.path.exists(video_path):
                extract_frames(video_path, output_folder, start_time, end_time)
            else:
                video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam05.mp4'.format(key))
                if os.path.exists(video_path):
                    extract_frames(video_path, output_folder, start_time, end_time)

        video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam03.mp4'.format(key))
        output_folder =os.path.join( final_path,'{}/exo3'.format(key),str(i))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            start_time = data['start_time']
            end_time = data['end_time']
            if os.path.exists(video_path):
                extract_frames(video_path, output_folder, start_time, end_time)
            else:
                video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam05.mp4'.format(key))
                if os.path.exists(video_path):
                    extract_frames(video_path, output_folder, start_time, end_time)

        video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam04.mp4'.format(key))
        output_folder =os.path.join( final_path,'{}/exo4'.format(key),str(i))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            start_time = data['start_time']
            end_time = data['end_time']
            if os.path.exists(video_path):
                extract_frames(video_path, output_folder, start_time, end_time)
            else:
                video_path = os.path.join(original_path,'{}/frame_aligned_videos/downscaled/448/cam05.mp4'.format(key))
                if os.path.exists(video_path):
                    extract_frames(video_path, output_folder, start_time, end_time)

current_directory = original_path
for name in names_noaction:
    for i in range(5):
        if i==0:
            video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/'.format(name))
            target_filename = "214-1.mp4" 
            ss = find_video_directory(video_path,target_filename)
            if ss is None:
                break
            video_path = os.path.join(video_path,ss)
            output_folder =os.path.join( final_path,'{}/ego'.format(name),str(0))
        else:
            if i==1:
                video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam01.mp4'.format(name))
                output_folder =os.path.join( final_path,'{}/exo'.format(name),str(0))
            elif i==2:
                video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam02.mp4'.format(name))
                if not os.path.exists(video_path):
                    video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam05.mp4'.format(name))
                    if os.path.exists(video_path): 
                        output_folder =os.path.join( final_path,'{}/exo{}'.format(name,i),str(0))
                else:
                    output_folder =os.path.join( final_path,'{}/exo{}'.format(name,i),str(0))
            elif i==3:
                video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam03.mp4'.format(name))
                if not os.path.exists(video_path):
                    video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam05.mp4'.format(name))
                    if os.path.exists(video_path): 
                        output_folder =os.path.join( final_path,'{}/exo{}'.format(name,i),str(0))
                else:
                    output_folder =os.path.join( final_path,'{}/exo{}'.format(name,i),str(0))
            else:
                video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam04.mp4'.format(name))
                if not os.path.exists(video_path):
                    video_path = os.path.join(current_directory,'{}/frame_aligned_videos/downscaled/448/cam05.mp4'.format(name))
                    if os.path.exists(video_path): 
                        output_folder =os.path.join( final_path,'{}/exo{}'.format(name,i),str(0))
                else:
                    output_folder =os.path.join( final_path,'{}/exo{}'.format(name,i),str(0))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            start_time = 0
            end_time = -1
            extract_frames(video_path, output_folder, start_time, end_time)
