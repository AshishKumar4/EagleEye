

cascade_path = './cv2/haarcascade_frontalface_alt2.xml'
def preprocess(images, margin=70, image_size=160):
    try:
        faceDetected = True
        aligned_images = []
        cascade = cv2.CascadeClassifier(cascade_path)
        for img in images:
            # print(filepath)
            if type(img) is list:
                img = np.array(img)
            img = to_rgb(img)
            try:
                faces = cascade.detectMultiScale(img,
                                                 scaleFactor=1.1,
                                                 minNeighbors=3)
                (x, y, w, h) = faces[0]
                #print(faces[0].dtype)
                cropped = img[y-margin//2:y+h+margin//2,
                              x-margin//2:x+w+margin//2, :]
                img = resize(cropped, (image_size, image_size), mode='reflect')
            except Exception as e:
                print("error in face detection")
                print(e)
                img = resize(img, (image_size, image_size), mode='reflect')
                faceDetected = False
            aligned_images.append(img)
            return np.array(aligned_images), faceDetected
    except Exception as e:
        print("Error in Preprocess ")
        print(e)
        return None

while True:
    frame = a.getFrame()
    n = np.array([frame])
    name, detect = c.classify(n)#preprocess(n)
    if not detect:
        print("No Face Detected!")
    #vec = c.embed(pre)
    #name = c.classify(pre, 'vec')
    print(name)
    cv2.imshow('Frame', frame)#pre[0])
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


def run():
    for i in processedStream():
        frame, vec, detect, name = i 
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break




def run():
    while True:
        frame = camera.getFrame()
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

#c.classify(c.embed(preprocess(np.array([cv2.imread('./images/Abhishek.jpg')]))), 'vec')
