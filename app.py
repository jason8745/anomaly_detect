
import keras_model
import common as com
from flask import Flask
from flask import request
from flask import jsonify
import json
import numpy as np
########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

app = Flask(__name__)


@app.route('/')
def hello_world():
    return '<h1>Welcome to use AI model</h1>'

@app.route('/detect',methods=['POST'])
def detect():
    output_dict={}
    output_dict['status']='Fail'
    if request.method == "POST":
        js =request.get_json(force=True)
        # print(type(js))
        raw_data =eval(js)
        # print(type(raw_data))         
        sound_data = np.array(raw_data['sound'])   
        # print(type(sound_data))
        # print(sound_data.shape)
        com.logger.info('model predict Now')            
        
        errors = np.mean(np.square(sound_data - model.predict(sound_data)), axis=1)
        com.logger.info('model Detect Over') 
        output_dict["errors"] = errors.tolist()
        output_dict["status"] = "Success"
        com.logger.info('detect status:{}'.format(output_dict["status"]))
    
    return jsonify(output_dict), 200





    
    


if __name__ == "__main__":
    
    model = keras_model.load_model('{model_path}/{model}'.format(model_path = param['model_directory'],
                                                                            model = param['model_name']))
    model.summary()
    com.logger.info("{model} load success".format(model = param['model_name']))
    app.run(debug=False, host='0.0.0.0', port=5000)