import cv2
import numpy as np

out_path = 'dashcam.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, 15.0, (320, 240))
for i in range(30):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i+1}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    writer.write(frame)
writer.release()
print('created', out_path)
