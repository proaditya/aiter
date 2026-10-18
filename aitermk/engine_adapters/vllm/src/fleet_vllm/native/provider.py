"""ctypes binding of a fleet MPK provider DSO (libaitermk_provider_gpt_oss_gfx950_fleet_mpk_<config>.so).

The DSO is the fleet persistent megakernel of one build configuration, exported by the fleet repository's
mirage.mpk.provider_export: every device address its init needs comes from a pointer table, and its task graph
is compiled in. ctypes releases the GIL during each call, so fleet_mpk_launch (which blocks for the whole
request) runs on a thread while another thread polls fleet_mpk_decode_progress.
"""

import ctypes as ct

ABI_VERSION = 1
NUM_META = 10


class Provider:
    def __init__(self, so_path):
        self.so_path = so_path
        self.lib = lib = ct.CDLL(so_path)

        def fn(name, restype, *argtypes):
            f = getattr(lib, name)
            f.restype = restype
            f.argtypes = list(argtypes)
            return f

        self.abi_version = fn("fleet_mpk_abi_version", ct.c_int)()
        if self.abi_version != ABI_VERSION:
            raise RuntimeError(
                f"{so_path}: fleet MPK ABI {self.abi_version}, this adapter speaks {ABI_VERSION}"
            )
        self.build_id = fn("fleet_mpk_build_id", ct.c_char_p)().decode()
        self.num_ptrs = fn("fleet_mpk_num_ptrs", ct.c_int)()
        name = fn("fleet_mpk_ptr_name", ct.c_char_p, ct.c_int)
        self.ptr_names = [name(i).decode() for i in range(self.num_ptrs)]
        vp = ct.POINTER(ct.c_void_p)
        self._init = fn(
            "fleet_mpk_init",
            ct.c_int,
            vp,
            ct.c_int,
            vp,
            ct.c_int,
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_longlong,
        )
        self.set_rope_tables = fn("fleet_mpk_set_rope_tables", None, ct.c_void_p, ct.c_void_p)
        self.set_max_seq_length = fn("fleet_mpk_set_max_seq_length", None, ct.c_int)
        self.init_request = fn("fleet_mpk_init_request", None)
        self.reset_decode_progress = fn("fleet_mpk_reset_decode_progress", None)
        self.launch = fn("fleet_mpk_launch", None, ct.c_void_p)
        self.decode_progress = fn("fleet_mpk_decode_progress", ct.c_int, ct.c_int)
        self.decode_token = fn("fleet_mpk_decode_token", ct.c_int, ct.c_int)
        self.finalize = fn("fleet_mpk_finalize", None)
        # Builds with MPK_HOST_STOP: end the running launch at its next 16-iteration check. Older DSOs lack it.
        self.request_stop = (
            fn("fleet_mpk_request_stop", None) if hasattr(lib, "fleet_mpk_request_stop") else None
        )

    def init(self, ptrs, meta, args):
        """ptrs: one device address per pointer (ptr_names order); meta: the 10 runtime buffers; args: init_args."""
        if len(ptrs) != self.num_ptrs or len(meta) != NUM_META:
            raise ValueError(
                f"fleet_mpk_init: {len(ptrs)}/{self.num_ptrs} pointers, {len(meta)}/{NUM_META} meta buffers"
            )
        p = (ct.c_void_p * len(ptrs))(*ptrs)
        m = (ct.c_void_p * len(meta))(*meta)
        rc = self._init(
            p,
            len(ptrs),
            m,
            len(meta),
            None,
            int(args["my_rank"]),
            int(args["num_workers"]),
            int(args["num_local_schedulers"]),
            int(args["num_remote_schedulers"]),
            int(args["max_seq_length"]),
            int(args["total_num_requests"]),
            int(args["eos_token_id"]),
        )
        if rc != 0:
            raise RuntimeError(f"fleet_mpk_init failed: {rc}")
