import cv2
import os
from deepface import DeepFace

output_path = './temp/'
os.makedirs(output_path, exist_ok=True)

vc = cv2.VideoCapture(0)
vc_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
vc_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
vc_fps = vc.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
outVideo = cv2.VideoWriter(output_path + 'example.mp4', fourcc, vc_fps, (int(vc_width), int(vc_height)))

i = 1
x, y, w, h = -1, -1, -1, -1
emo = ''
color = (0, 0, 255)

while True:
    ret, frame = vc.read()
    if not ret:
        break
    else:
        if (i % vc_fps == 0):
            i = 0
            cam_origin = output_path + 'cam_origin' + '.jpg'
            try:
                cv2.imwrite(cam_origin, frame)
            except:
                ex = Exception('save image failed: ' + cam_origin)
                raise ex
            
            analyze_res = DeepFace.analyze(img_path = cam_origin, actions = ['emotion'], enforce_detection = False, silent = True)

            x, y = analyze_res[0]['region']['x'], analyze_res[0]['region']['y']
            w, h = analyze_res[0]['region']['w'], analyze_res[0]['region']['h']
            emo = analyze_res[0]['dominant_emotion']

        if (w == vc_width and h == vc_height) or (w == -1 and h == -1):
            pass
        else:
            cv2.rectangle(frame, (x ,y), (x + w, y + h), color, 1, 8)
            cv2.putText(frame, emo, (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)                
        cv2.imshow('video',frame)
        outVideo.write(frame)
    i = i + 1
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
 
vc.release()
cv2.destroyAllWindows()