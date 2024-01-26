import lunar_tools as lt
import time
import cv2

# Create server and client
server = lt.ZMQPairEndpoint(is_server=True, ip='10.20.16.145', port='5559')

while True:
    time.sleep(0.1)
    image = server.get_img()
    if image is None:
        continue
    print("received image")
    # Display the image
    cv2.imshow('Image', image)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
