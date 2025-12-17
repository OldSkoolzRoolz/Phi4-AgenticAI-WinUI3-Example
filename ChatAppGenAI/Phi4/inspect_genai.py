# inspect_genai_api.py
import importlib, inspect, json
m = importlib.import_module("onnxruntime_genai")
print("Module file:", getattr(m, "__file__", "<built-in>"))
names = sorted([n for n in dir(m) if not n.startswith("_")])
print("Top-level names:", names)

# show callables that might create sessions
callables = []
for n in names:
    obj = getattr(m, n)
    if inspect.isfunction(obj) or inspect.isclass(obj):
        callables.append((n, inspect.signature(obj) if inspect.isfunction(obj) else "class"))
print("Callables (sample):")
for c in callables[:60]:
    print(" ", c)