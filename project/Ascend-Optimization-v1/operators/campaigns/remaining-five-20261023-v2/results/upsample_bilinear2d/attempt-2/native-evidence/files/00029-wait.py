import json
import os
import time

d = os.environ.get("KGS_DEBUG_ARTIFACTS", "")
if not (d and os.path.isdir(d)):
    d = os.getcwd()

time.sleep(170)
p = os.path.join(d, "wait.json")
with open(p, "w") as fh:
    json.dump({"slept": 170}, fh)
print("done", flush=True)
