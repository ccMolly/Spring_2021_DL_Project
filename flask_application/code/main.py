import os
import time

from flask import Flask
from flask import request
from flask import render_template
from inference import inference

app = Flask(__name__, static_folder='user_imgs')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Receive image sent from user
        image_name = f'{int(time.time())}.jpg'
        image_file = request.files['image']
        image_loc = os.path.join("user_imgs", image_name)
        image_file.save(image_loc)
        # Receive prune_type sent from user: none / channel / filter
        prune_type = request.form.get('prune_type')
        # Receive model_type sent from user: vgg / resnet
        model_type = request.form.get('model_type')

        # Call inference method and get back predicted class name, model inference type and prune_type/model_type selected by user
        class_name, inference_time, s, prune_type, model_type = inference(image_loc, prune_type, model_type)
        
        # Render html to show results
        return render_template('index.html', res=class_name, image_name=image_name, 
            inference_time=inference_time, model_summary=s, prune_type=prune_type, model_type=model_type)

    return render_template('index.html', res=None, image_name=None, 
        inference_time=None, model_summary=None, prune_type=None, model_type=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
