# run_genai_simple.py
import os, sys
repo_dir = os.path.abspath(".")
prompt = " ".join(sys.argv[1:]) or "Hello world"

import onnxruntime_genai as ortg

# Load model and tokenizer using the GenAI API
Model = ortg.Model
Tokenizer = ortg.Tokenizer

# Try common load patterns
model = None
tokenizer = None
if hasattr(Model, "from_pretrained"):
    model = Model.from_pretrained(repo_dir)
elif hasattr(Model, "load"):
    model = Model.load(repo_dir)
else:
    model = Model(repo_dir)

if hasattr(Tokenizer, "from_pretrained"):
    tokenizer = Tokenizer.from_pretrained(repo_dir)
else:
    tokenizer = Tokenizer(repo_dir)

# Try generation via model or Generator
if hasattr(model, "generate"):
    out = model.generate(prompt)
    print(out)
    sys.exit(0)

# fallback: try Generator class if present
Generator = getattr(ortg, "Generator", None)
if Generator is not None:
    try:
        gen = Generator(model=model, tokenizer=tokenizer)
    except Exception:
        gen = Generator(model)
    out = gen.generate(prompt)
    print(out)
    sys.exit(0)

print("Could not call generate automatically. Model methods:", [m for m in dir(model) if not m.startswith('_')][:80])