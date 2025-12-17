python - <<'PY'
import onnxruntime as ort, os, json
sess = ort.InferenceSession("Phi4Model\model.onnx")
print("Inputs:")
for i in sess.get_inputs(): print(i.name, i.shape, i.type)
print("Outputs:")
for o in sess.get_outputs(): print(o.name, o.shape, o.type)
PY