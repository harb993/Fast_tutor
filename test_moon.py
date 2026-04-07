from moonshine_onnx import MoonshineOnnxModel 
stt = MoonshineOnnxModel(model_name="moonshine/tiny") 
print([m for m in dir(stt) if not m.startswith('_')])
