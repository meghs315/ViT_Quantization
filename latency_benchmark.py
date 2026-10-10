import time, copy, torch
from transformers import ViTForImageClassification

dev = "cuda"
base = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(dev).eval()

def bench(model, x, iters=50, warmup=10):
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(dev).eval()
        if dtype == torch.float16:
            model = model.half()
        for bs in (1, 8, 32):
            x = torch.randn(bs, 3, 224, 224, device=dev, dtype=dtype)
            lat, thr, mem = bench(model, x)
            print(f"{name} bs={bs}: {lat*1000:.1f} ms, {thr:.0f} img/s, {mem:.0f} MiB")
        del model
        torch.cuda.empty_cache()