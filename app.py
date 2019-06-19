from flask import Flask, render_template, request
from werkzeug import secure_filename
import tensorflow as tf
from keras.models import load_model
import numpy as np
# import pytesseract
#from sklearn.metrics import accuracy_score,confusion_matrix
from keras.preprocessing import image
# import  Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
app = Flask(__name__)
global graph
graph = tf.get_default_graph()
model=load_model('G:\\20.h5')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def upload():
    return render_template('upload.html')



@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, '/')
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("file"):
            print(file)
            filename = file.filename
            destination = "".join([target, filename])
            print(destination)
            file.save(destination)
        # f = request.files['file']
        # f.save(secure_filename(f.filename))
        # print(f)
        #f.save('G:\result')
        test_image = image.load_img('/'+filename,target_size=(224, 224, 3))
        print(test_image)
        # text = pytesseract.image_to_string(test_image)
        # print(text)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        with graph.as_default():
            a = model.predict(test_image)
        print(a)
        print(np.argmax(a))
        str1 = str(np.argmax(a))a
        res = int(str1)
        print('file uploaded successfully')

        if(res == 0):
            return render_template('0.html')
        if (res == 1):
            return render_template('1.html')
        if (res == 2):
            return render_template('2.html')
        if (res == 3):
            return render_template('3.html')
        if (res == 4):
            return render_template('41.html')
        if (res == 5):
            return render_template('3.html')
        if (res == 6):
            return render_template('6.html')
        if (res == 7):
            return render_template('7.html')
        if (res == 8):
            return render_template('3.html')
        if (res == 9):
            return render_template('9.html')
        if (res == 10):
            return render_template('10.html')
        if (res == 11):
            return render_template('3.html')
        if (res == 12):
            return render_template('12.html')
        if (res == 13):
            return render_template('3.html')
        if (res == 14):
            return render_template('14.html')
        if (res == 15):
            return render_template('3.html')
        if (res == 16):
            return render_template('16.html')
        if (res == 17):
            return render_template('3.html')
        if (res == 18):
            return render_template('18.html')
        if (res == 19):
            return render_template('3.html')



        #return render_template('3.html')


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)
