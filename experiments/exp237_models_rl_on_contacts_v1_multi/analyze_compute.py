import re, glob, numpy as np
from collections import defaultdict
ANSI = re.compile(r"\x1b\[[0-9;]*m")
KEYS = ["timing/generate", "timing/fwd_logprobs_values_reward", "timing/policy_train",
        "timing/sync_weights", "timing/step", "trainer/tokens_per_second_per_gpu",
        "generate/avg_assistant_tokens"]
agg = defaultdict(list)
for f in glob.glob("/home/ubuntu/exp237_logs/exp237_*.log"):
    for raw in open(f, errors="replace"):
        l = ANSI.sub("", raw)
        for k in KEYS:
            m = re.search("'" + re.escape(k) + "': '([0-9.]+)'", l)
            if m:
                agg[k].append(float(m.group(1)))
print("%-40s %6s %8s %8s" % ("", "n", "median", "mean"))
for k in KEYS:
    v = np.array(agg[k])
    if len(v):
        print("%-40s %6d %8.1f %8.1f" % (k, len(v), np.median(v), v.mean()))
g = np.median(agg["timing/generate"]); fw = np.median(agg["timing/fwd_logprobs_values_reward"])
tr = np.median(agg["timing/policy_train"]); sy = np.median(agg["timing/sync_weights"])
st = np.median(agg["timing/step"]); resp = np.median(agg["generate/avg_assistant_tokens"])
print("\nphases sum to %.1fs against a %.1fs step -> the pipeline is SERIAL" % (g+fw+tr+sy, st))
used = 6*g + 2*fw + 1*tr + 8*sy; avail = 8*st
print("GPU-seconds used %.0f of %.0f -> node utilisation %.0f%%" % (used, avail, 100*used/avail))
T = 64*(resp+380); N = 1.47e9; fl = 8*N*T
print("\ntraining: %.0fk tokens/step, %.0f TFLOP, %.0fs on ONE A100" % (T/1e3, fl/1e12, tr))
print("  %.0f TFLOP/s = %.0f%% MFU during the training phase" % (fl/tr/1e12, 100*fl/tr/312e12))
print("  %.0f%% MFU amortised over the step (that GPU) | %.1f%% node-wide" % (
    100*fl/(st*312e12), 100*fl/(8*st*312e12)))
gt = 64*resp
print("\ngeneration: %.0fk tokens in %.0fs over 6 engines = %.0f tok/s (%.0f/engine), ~%.0f seqs/engine"
      % (gt/1e3, g, gt/g, gt/g/6, 64/6))
print("  decode FLOPs %.1f TFLOP/s over 6 GPUs = %.1f%% MFU -- decode is bandwidth-bound, not compute-bound"
      % (2*N*gt/g/1e12, 100*2*N*gt/g/(6*312e12)))
