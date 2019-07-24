# Steggy.(Smart)Chair
<p>A smart wheelchair that uses machine learning to identify and follow people around. The midline of the image is compared to the midline of the person's boundary box and the wheelchair's continous servos will adjust themselves to center the two midlines. Includes a web-app with buttons for manual control and the code to have the wheelchair move automatically without any input.</p>

<p>Demo: https://vimeo.com/349996145 </p>

# How it works
<p>Steggy.Chair utilized a Flask web server to connect our machine learning system to the Raspberry Pi. The Raspberry Pi controls the servos and the webcam. The webcam will take a picture of the person it is following and send it to the web server. The web server runs the ML and will detect the person. This is where the more complicated part of the program. The program will return the midline of the person detected, and it will return the midline of the image. The point here is to center the person in the image, so Steggy.Chair figures out which way to move the wheelchair to center the person on the image. The Pi does this by moving the wheelchair until the person is considered centered. Then, Steggy.Chair moves forward for 3 seconds and repeats the centering process again.</p>

