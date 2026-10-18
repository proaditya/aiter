"""Import a fleet MPK provider export as a build configuration.

usage: import_export.py EXPORT_DIR CONFIG_DIR [--check REPO_CONFIG_DIR]

EXPORT_DIR is a fleet MPK_PROVIDER_EXPORT directory (its src/ holds kernel.hip, task_graph.json, compile_cmd.json,
env.json, provider.json, init_args.json). Writes CONFIG_DIR/{kernel.hip, task_graph.json, compile_flags.txt,
provider.json, init_args.json, env.json}: the compile command minus its input, output, include and link
arguments (the build adds its own), the exporter's environment minus its export controls.

--check: fail unless kernel.hip, task_graph.json and compile_flags.txt equal REPO_CONFIG_DIR's apart from the
build id. The vendored headers (include/third_party/fleet) match the repository's sources, so an export from
another fleet commit must not be built against them.
"""

import argparse
import json
import os
import re
import shutil
import sys

DROP_EXACT = {"-x", "hip", "-shared", "-fPIC", "--save-temps"}
EXPORT_CONTROLS = ("MPK_PROVIDER_EXPORT", "MPK_PROVIDER_EXPORT_NODATA", "MPK_WRAP")
CHECKED = ("kernel.hip", "task_graph.json", "compile_flags.txt")


def flags_of(cmd):
    """The compile command's flags without the compiler, its input, output, include and link arguments."""
    out, skip = [], False
    for i, a in enumerate(cmd):
        if i == 0 or skip:
            skip = False
            continue
        if a == "-o":
            skip = True
        elif a in DROP_EXACT or a.endswith((".cu", ".hip")) or a.startswith(("-I", "-L", "-l")):
            continue
        else:
            out.append(a)
    return out


def norm(text):
    """The text with its build id replaced, for comparing two exports of the same build."""
    return re.sub(r"fleet-mpk-\d{8}T\d{6}Z", "BUILD", text)


def import_export(export_dir, config_dir):
    """Write config_dir from export_dir; return (build id, flags)."""
    s = os.path.join(export_dir, "src")
    os.makedirs(config_dir, exist_ok=True)
    for f in ("kernel.hip", "task_graph.json", "provider.json", "init_args.json"):
        shutil.copy(os.path.join(s, f), os.path.join(config_dir, f))
    with open(os.path.join(s, "env.json")) as f:
        env = {k: v for k, v in json.load(f).items() if k not in EXPORT_CONTROLS}
    with open(os.path.join(config_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=0)
    with open(os.path.join(s, "compile_cmd.json")) as f:
        flags = flags_of(json.load(f))
    with open(os.path.join(config_dir, "compile_flags.txt"), "w") as f:
        f.write("\n".join(flags) + "\n")
    with open(os.path.join(s, "provider.json")) as f:
        build_id = json.load(f)["build_id"]
    return build_id, flags


def differing(config_dir, repo_config_dir):
    """The checked files whose content differs between the two configurations, apart from the build id."""

    def read(d, f):
        with open(os.path.join(d, f)) as fh:
            return norm(fh.read())

    return [f for f in CHECKED if read(config_dir, f) != read(repo_config_dir, f)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("config_dir")
    ap.add_argument("--check")
    a = ap.parse_args(argv)
    build_id, flags = import_export(a.export_dir, a.config_dir)
    ndefs = sum(1 for f in flags if f.startswith("-D"))
    print(
        f"imported {a.export_dir} -> {a.config_dir}: build {build_id}, {len(flags)} flags ({ndefs} -D)"
    )
    if a.check:
        bad = differing(a.config_dir, a.check)
        if bad:
            sys.exit(
                f"export differs from {a.check} in {bad}: built from another fleet commit or configuration"
            )
        print(f"matches {a.check} apart from the build id")


if __name__ == "__main__":
    main()
