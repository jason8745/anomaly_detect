import librosa
import numpy as np
import common as com
import json

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

class Preprocess():
    def __init__(self,file_path):
        self.path = file_path
        self.data =None
        self.convert()
        self.jsdata =self.tojson()

    def convert(self):
        """
        Convert .wav data to vector array
        """        


        self.data = com.file_to_vector_array(self.path,
                    n_mels=param["feature"]["n_mels"],
                    frames=param["feature"]["frames"],
                    n_fft=param["feature"]["n_fft"],
                    hop_length=param["feature"]["hop_length"],
                    power=param["feature"]["power"])
    def tojson(self):
        """
        flask just can read json
        and numpy array need to be transform to list
        1. ndarray -->list
        2.dict dump to json
        """
        temp = {}
        temp['sound'] = self.data.tolist()
        # print(type(temp))
        temp = json.dumps(temp)
        # print(type(temp))

        return  temp         

