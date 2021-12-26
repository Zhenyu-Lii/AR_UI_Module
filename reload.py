from keras.models import model_from_json

def __call__(filename):
    if (filename == None):
        print("需要输入文件名")
        return
    else:
        # 要重新加载的keras模型
        file_name = filename
        json_path = "checkpoints/"+file_name+".json"
        h5_path = "./checkpoints/"+file_name+".h5"
        print("获得了json和h5文件")
        # 从JSON文件中加载模型
        with open(json_path, 'r') as file:
            model_json = file.read()

        # 加载模型权重
        new_model = model_from_json(model_json)
        new_model.load_weights(h5_path)

        new_model.save("./kerasModel/"+file_name+".h5")
        print("转换完成")

if __name__ == '__main__':
    default_source = "LSTM_LIBROSA"
    __call__(default_source)
else:
    __call__()