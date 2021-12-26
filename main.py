from keras.models import model_from_json
import tensorflow as tf

def reload(filename):
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

def convertToLite(filename):
    source_path = "kerasModel/" + filename + ".h5"
    save_path = "tfLite/" + filename + ".tflite"
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(source_path)
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

if __name__ == '__main__':
    print("作为主程序\n")
    name_list = ["CNN1D_OPENSMILE_IS10", "LSTM_LIBROSA", "LSTM_OPENSMILE", "LSTM_OPENSMILE_IS09",
                 "LSTM_OPENSMILE_IS10", "LSTM_OPENSMILE_IS11", "LSTM_OPENSMILE_IS16"]
    for default_file in name_list:
        reload(default_file)
        convertToLite(default_file)
    print("转换完成\n")
