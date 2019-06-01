import cv2
import requests
import time
import json
import pigpio

camera_port = 0
ramp_frames = 30

ip = "http://64.187.241.17:5000/image_proc"
imgsrc = "/home/pi/Desktop/Wheelchair/Images/img.jpg"

print("Camera Init")
camera = cv2.VideoCapture(camera_port)

threshold = 200

pi = pigpio.pi()
pi.set_mode(12, pigpio.OUTPUT)
pi.set_mode(18, pigpio.OUTPUT)
pi.set_mode(4, pigpio.OUTPUT)
pi.set_mode(5, pigpio.OUTPUT)

def stop():
    pi.set_servo_pulsewidth(12, 0)
    pi.set_servo_pulsewidth(18, 0)
    pi.set_servo_pulsewidth(4, 0)
    pi.set_servo_pulsewidth(5, 0)

def get_image ():
    retval, im = camera.read()
    return im
x = 0;
def right(difference):
    pi.set_servo_pulsewidth(12, 2100)
    pi.set_servo_pulsewidth(18, 2100)
    pi.set_servo_pulsewidth(4, 2100)
    pi.set_servo_pulsewidth(5, 2100)
    time.sleep(difference/275)
    stop()
def left(difference):
    pi.set_servo_pulsewidth(12, 600)
    pi.set_servo_pulsewidth(18, 600)
    pi.set_servo_pulsewidth(4, 600)
    pi.set_servo_pulsewidth(5, 600)
    time.sleep(difference/275)
    stop()
def forward():
    pi.set_servo_pulsewidth(18, 2500)
    pi.set_servo_pulsewidth(5, 2500)
    pi.set_servo_pulsewidth(4, 600)
    pi.set_servo_pulsewidth(12, 600)
    time.sleep(1)
    stop()

try:
    while True:
        for i in range(ramp_frames):
            temp = get_image()
        print("Taking image")
        camera_capture = get_image()
        print("Done")

        print("Saving File")
        file = imgsrc
        cv2.imwrite(file, camera_capture)
        print("Done")

        print("Sending Image")
        r = requests.get(ip, files=dict(image=open(imgsrc, 'rb')))
        print("Done")
        try:
            data = json.loads(r.text)
            print(data)
            print(data['prediction'])
            print(float(data['image_midline']))
            print(float(data['midline']))
            diff = float(data['image_midline']) - float(data['midline'])
            print("diff is {}".format(diff))
            
            if(diff > threshold):
                left(diff)
            elif(diff < -threshold):
                right(abs(diff))
            else:
                forward()
        except json.decoder.JSONDecodeError:
            print("No Person Found")
except KeyboardInterrupt:
    
    pi.set_servo_pulsewidth(12, 0)
    pi.set_servo_pulsewidth(18, 0)
    pi.set_servo_pulsewidth(4, 0)
    pi.set_servo_pulsewidth(5, 0)
