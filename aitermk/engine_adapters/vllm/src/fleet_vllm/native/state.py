"""Rebuild a fleet MPK provider's device state from its export (state/manifest.json + alloc_*.bin).

The exporter (fleet repository, mirage.mpk.provider_export) records every device allocation the kernel's
pointers land in, byte for byte: torch caching-allocator segments (one device each) and HIP VMM buffers whose
region d sits on the first partition of AID d (DPX/CPX, MPK_MULTI_DEV). This module recreates each allocation
of the same size on the same partitions, loads its bytes, and translates every recorded (allocation, offset)
into a device address. Offsets inside an allocation are preserved, so layouts the kernel derives from them
(placement arena halves, split experts, ring copies) stay valid.
"""

import ctypes as ct
import json
import os
import time

import torch

_HIP = None
_GiB = float(1 << 30)


def hip():
    global _HIP
    if _HIP is None:
        _HIP = ct.CDLL("libamdhip64.so")
    return _HIP


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: HIP error {rc}")


class _Loc(ct.Structure):
    _fields_ = [("type", ct.c_int), ("id", ct.c_int)]


class _AllocFlags(ct.Structure):
    _fields_ = [
        ("compressionType", ct.c_ubyte),
        ("gpuDirectRDMACapable", ct.c_ubyte),
        ("usage", ct.c_ushort),
    ]


class _Prop(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("requestedHandleType", ct.c_int),
        ("location", _Loc),
        ("win32HandleMetaData", ct.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _Access(ct.Structure):
    _fields_ = [("location", _Loc), ("flags", ct.c_int)]


def _vmm_device(a, d):
    """HIP device holding region d of a VMM allocation: the first partition of its AID."""
    aid = (a["domains"] - 1 - d) if a["swap"] else d
    return aid * (a["mdev"] // a["domains"])


def vmm_halves(nbytes, domains, swap, mdev):
    """nbytes of device memory, one VA, region d (nbytes / domains) on the first partition of AID d."""
    h = hip()
    region = nbytes // domains
    if region * domains != nbytes or region % (2 << 20):
        raise ValueError(f"vmm_halves({nbytes}, {domains})")
    va = ct.c_void_p()
    _check(
        h.hipMemAddressReserve(
            ct.byref(va), ct.c_size_t(nbytes), ct.c_size_t(2 << 20), None, ct.c_ulonglong(0)
        ),
        f"hipMemAddressReserve({nbytes})",
    )
    prop = _Prop()
    prop.type = 1  # hipMemAllocationTypePinned
    prop.location.type = 1  # hipMemLocationTypeDevice
    cur = ct.c_int()
    _check(h.hipGetDevice(ct.byref(cur)), "hipGetDevice")
    rec = {"domains": domains, "swap": swap, "mdev": mdev}
    try:
        for d in range(domains):
            prop.location.id = _vmm_device(rec, d)
            # HIP 7.2's hipMemCreate uses the current device, not location.id
            _check(h.hipSetDevice(prop.location.id), f"hipSetDevice({prop.location.id})")
            handle = ct.c_void_p()
            _check(
                h.hipMemCreate(
                    ct.byref(handle), ct.c_size_t(region), ct.byref(prop), ct.c_ulonglong(0)
                ),
                f"hipMemCreate of VMM region {d} ({region} B)",
            )
            _check(
                h.hipMemMap(
                    ct.c_void_p(va.value + d * region),
                    ct.c_size_t(region),
                    ct.c_size_t(0),
                    handle,
                    ct.c_ulonglong(0),
                ),
                f"hipMemMap of VMM region {d}",
            )
    finally:
        h.hipSetDevice(cur.value)
    acc = (_Access * mdev)()
    for k in range(mdev):
        acc[k].location.type = 1
        acc[k].location.id = k
        acc[k].flags = 3  # hipMemAccessFlagsProtReadWrite
    _check(h.hipMemSetAccess(va, ct.c_size_t(nbytes), acc, ct.c_size_t(mdev)), "hipMemSetAccess")
    return va.value


def _hip_malloc(nbytes, device):
    h = hip()
    cur = ct.c_int()
    _check(h.hipGetDevice(ct.byref(cur)), "hipGetDevice")
    p = ct.c_void_p()
    try:
        _check(h.hipSetDevice(device), f"hipSetDevice({device})")
        rc = h.hipMalloc(ct.byref(p), ct.c_size_t(nbytes))
    finally:
        h.hipSetDevice(cur.value)
    if rc != 0 or not p.value:
        raise RuntimeError(f"hipMalloc({nbytes}) on device {device} failed: HIP error {rc}")
    return p.value


def _pci_bus_id(device):
    buf = ct.create_string_buffer(64)
    _check(
        hip().hipDeviceGetPCIBusId(buf, ct.c_int(len(buf)), ct.c_int(device)),
        "hipDeviceGetPCIBusId",
    )
    return buf.value.decode()


def read_manifest(state_dir):
    """The export's manifest, with init_args and env filled in from src/ when the manifest lacks them."""
    with open(os.path.join(state_dir, "manifest.json")) as f:
        man = json.load(f)
    src = os.path.join(os.path.dirname(os.path.abspath(state_dir)), "src")
    for key, name in (("init_args", "init_args.json"), ("env", "env.json")):
        if not man.get(key):
            with open(os.path.join(src, name)) as f:
                man[key] = json.load(f)
    return man


def preflight(man, state_dir, log=print):
    """Fail in seconds, before loading the state, unless this process can hold it.

    Checks that every allocation file is complete, that the visible devices are the build's partitions of one
    gfx950 GPU with the CUs its launch fills, and that each has room for its share of the state.
    """
    allocs = man["allocations"]
    need = {}  # device -> bytes
    for a in allocs:
        if a["kind"] == "vmm":
            for d in range(a["domains"]):
                dev = _vmm_device(a, d)
                need[dev] = need.get(dev, 0) + a["size"] // a["domains"]
        else:
            need[a["device"]] = need.get(a["device"], 0) + a["size"]
        path = os.path.join(state_dir, a["file"])
        if not os.path.isfile(path):
            raise RuntimeError(f"{path} is missing: the export is incomplete")
        if os.path.getsize(path) != a["size"]:
            raise RuntimeError(
                f"{path} has {os.path.getsize(path)} B, the manifest records {a['size']} B: "
                "the export is incomplete or from another build"
            )
    mdev = max([a["mdev"] for a in allocs if a["kind"] == "vmm"] + [1])
    ndev = max(mdev, max(need) + 1)
    visible = torch.cuda.device_count()
    if visible < ndev:
        raise RuntimeError(
            f"the provider needs {ndev} HIP devices (the DPX partitions of one MI355X), {visible} visible: "
            "set HIP_VISIBLE_DEVICES to the GPU's partitions"
        )
    args = man["init_args"]
    per_device = (int(args["num_workers"]) + int(args["num_local_schedulers"])) // mdev
    for d in range(ndev):
        p = torch.cuda.get_device_properties(d)
        arch = getattr(p, "gcnArchName", "") or p.name
        if not arch.startswith("gfx950"):
            raise RuntimeError(
                f"HIP device {d} is {arch}: the provider is built for gfx950 (MI355X)"
            )
        if p.multi_processor_count != per_device:
            raise RuntimeError(
                f"HIP device {d} has {p.multi_processor_count} CUs, the provider's launch fills {per_device} per "
                f"device: run the GPU in DPX+NPS2 and use its {mdev} partitions"
            )
    buses = [_pci_bus_id(d) for d in range(ndev)]
    if len({b.rsplit(".", 1)[0] for b in buses}) != 1:
        raise RuntimeError(
            f"HIP devices {list(range(ndev))} ({', '.join(buses)}) are not the partitions of one GPU: "
            "set HIP_VISIBLE_DEVICES to one GPU's partitions"
        )
    for dev, nbytes in sorted(need.items()):
        free, _ = torch.cuda.mem_get_info(dev)
        if nbytes + (1 << 30) > free:
            raise RuntimeError(
                f"HIP device {dev} needs {nbytes / _GiB:.1f} GiB for the provider state, {free / _GiB:.1f} GiB "
                "are free: is another process using the GPU?"
            )
    log(
        f"fleet_vllm: preflight ok: {ndev} partitions of {buses[0].rsplit('.', 1)[0]}, {per_device} CUs each, "
        + ", ".join(f"{need[d] / _GiB:.1f} GiB on device {d}" for d in sorted(need))
    )


class ProviderState:
    def __init__(self, state_dir, chunk=256 << 20, log=print):
        self.dir = state_dir
        self.man = man = read_manifest(state_dir)
        preflight(man, state_dir, log=log)
        self.init_args = man["init_args"]
        self._env = man["env"]
        self.build_id = man.get("build_id")
        self.base, self._torch, self._keep = {}, {}, []
        h = hip()
        host = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
        hp, hv = host.data_ptr(), memoryview(host.numpy())
        t0, total = time.time(), 0
        for a in man["allocations"]:
            kind, size = a["kind"], a["size"]
            if kind == "torch":
                t = torch.empty(size, dtype=torch.uint8, device=f"cuda:{a['device']}")
                self._torch[a["id"]] = t
                base = t.data_ptr()
            elif kind == "vmm":
                base = vmm_halves(size, a["domains"], a["swap"], a["mdev"])
            elif kind == "hipmalloc":
                base = _hip_malloc(size, a["device"])
            else:
                raise RuntimeError(f"unknown allocation kind {kind!r}")
            self.base[a["id"]] = base
            with open(os.path.join(state_dir, a["file"]), "rb", buffering=0) as f:
                done = 0
                while done < size:
                    n = f.readinto(hv[: min(chunk, size - done)])
                    if not n:
                        raise RuntimeError(f"{a['file']}: short file at {done} of {size} B")
                    _check(
                        h.hipMemcpy(ct.c_void_p(base + done), ct.c_void_p(hp), ct.c_size_t(n), 1),
                        f"hipMemcpy H2D into allocation {a['id']}",
                    )
                    done += n
            total += size
        self._keep.append(host)
        log(
            f"fleet_vllm: provider state {len(man['allocations'])} allocations, {total / _GiB:.2f} GiB loaded "
            f"in {time.time() - t0:.0f} s from {state_dir}"
        )

    def addr(self, alloc, offset):
        return self.base[alloc] + offset

    def ptrs(self):
        return [
            self.addr(p["alloc"], p["offset"])
            for p in sorted(self.man["ptrs"], key=lambda p: p["idx"])
        ]

    def meta(self):
        return [self.addr(m["alloc"], m["offset"]) for m in self.man["meta"]]

    def rope(self):
        r = self.man["rope"]
        return self.addr(r["cos"]["alloc"], r["cos"]["offset"]), self.addr(
            r["sin"]["alloc"], r["sin"]["offset"]
        )

    def env(self):
        """The build's MPK_* runtime environment, drafter pointer lists relocated to this process."""
        env = {k: v for k, v in self._env.items() if k != "MPK_PROVIDER_EXPORT"}
        for k, lst in self.man.get("env_ptrs", {}).items():
            env[k] = ",".join(f"{self.addr(a, o):x}" for a, o in lst)
        return env

    def view(self, rec):
        """A torch tensor over a recorded buffer (torch-segment allocations only)."""
        t = self._torch.get(rec["alloc"])
        if t is None:
            raise RuntimeError(f"{rec.get('name')} is not in a torch allocation")
        dt = getattr(torch, rec["dtype"])
        flat = t[rec["offset"] : rec["offset"] + rec["nbytes"]].view(dt)
        return flat.as_strided(rec["shape"], rec["stride"])

    def meta_view(self, name):
        for m in self.man["meta"]:
            if m["name"] == name:
                return self.view(m)
        raise KeyError(name)

    def reset_views(self):
        return [self.view(r) for r in self.man.get("reset", [])]
