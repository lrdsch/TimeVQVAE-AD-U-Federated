"""Verifica una riga sola: che precisione userebbe questo host, con e senza il pin."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from pipeline.federated import _amp_dtype  # noqa: E402

name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else ("-", "-")
os.environ.pop("FEDVQ_AMP", None)
auto = _amp_dtype()
os.environ["FEDVQ_AMP"] = "fp16"
pin = _amp_dtype()
print(f"  {name} sm_{cap[0]}{cap[1]}   auto -> {auto}   pin(fp16) -> {pin}")
