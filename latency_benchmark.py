import time, copy, torch
from transformers import ViTForImageClassification

dev = "cuda"
base = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(dev).eval()

def bench(model, x, iters=50, warmup=10):
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            model(pixel_values=x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t = time.perf_counter()
            model(pixel_values=x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    times.sort()
    med = times[len(times) // 2]
    return med, x.shape[0] / med, torch.cuda.max_memory_allocated() / 2**20

fp16 = copy.deepcopy(base).half()
for bs in (1, 8, 32):
    x = torch.randn(bs, 3, 224, 224, device=dev)
    for name, m, inp in (("fp32", base, x), ("fp16", fp16, x.half())):
        lat, thr, mem = bench(m, inp)
        print(f"{name} bs={bs}: {lat*1000:.1f} ms, {thr:.0f} img/s, {mem:.0f} MiB")