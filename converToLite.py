import tensorflow as tf
def __call__(filename):
    source_path = "kerasModel/" + filename + ".h5"
    save_path = "tfLite/" + filename + ".tflite"
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(source_path)
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

if __name__ == '__main__':
    default_source = "LSTM_OPENSMILE_IS11"
    __call__(default_source)
else:
    __call__()
