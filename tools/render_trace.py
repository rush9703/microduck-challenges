"""Render an Arena replay (a stored evaluation's trace) to an mp4, offscreen.

The trace is what the Arena's evaluator recorded for the median run of a checkpoint: per 50 Hz
frame, the trunk's free joint (7) then the 14 servo angles. It is replayed kinematically on the
event's own course, with the robot's visual meshes put back (the Arena strips them for physics).

    MUJOCO_GL=egl uv run --frozen --group workshop --with "imageio[ffmpeg]" python tools/render_trace.py \
        logs/rsl_rl/hill_climb_1m5/<run>/workshop/model_1000.json out.mp4
"""

from __future__ import annotations

import base64
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def _render_model(event: str):
    """The event's course with the robot's visual geoms kept; falls back to the physics model."""
    from arena import course as C
    from arena.assets import ASSET_ROOT
    from arena.manifest import load_event

    manifest = load_event(event)
    try:
        rig_dir, xml_name = C.RIG_XML[manifest.rig]
        robot_dir = ASSET_ROOT / "robot" / rig_dir
        root = ET.parse(robot_dir / xml_name).getroot()
        root.find("compiler").set("meshdir", "")
        assets = {m.get("file"): (robot_dir / "assets" / m.get("file")).read_bytes()
                  for m in root.find("asset").findall("mesh")}
        C._inject_course(root, manifest)
        C._inject_keyframe(root, manifest)
        wb = root.find("worldbody")
        ET.SubElement(wb, "light", {"pos": "0.5 0 2", "dir": "0 0 -1", "directional": "true"})
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    except Exception as exc:  # noqa: BLE001
        print(f"[render] visual model failed ({exc}); using the physics model", file=sys.stderr)
        model = C.compile_course(manifest)
    return model


def render_frames(trace: np.ndarray, event: str, width=640, height=480, every=2):
    """trace: (frames, 21). Yields RGB frames at 50/every fps, chase camera on the trunk."""
    model = _render_model(event)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    trunk = model.body("trunk_base").id
    free = int(model.jnt_qposadr[model.body_jntadr[trunk]])
    from arena.constants import JOINT_NAMES

    qadr = np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = trunk
    cam.distance, cam.azimuth, cam.elevation = 0.7, 135.0, -20.0
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for frame in trace[::every]:
            data.qpos[free:free + 7] = frame[0:7]
            data.qpos[qadr] = frame[7:21]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            yield renderer.render()


def render_body(body: dict, event: str, out: Path, fps_every=2) -> Path:
    from arena.constants import TRACE_WIDTH

    trace = np.frombuffer(base64.b64decode(body["trace_b64"]), dtype="<f4").reshape(-1, TRACE_WIDTH)
    import imageio.v2 as imageio

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=50 // fps_every, codec="libx264", quality=7, macro_block_size=8) as w:
        for img in render_frames(trace, event, every=fps_every):
            w.append_data(img)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    event = sys.argv[3] if len(sys.argv) > 3 else "hill-climb-1m5"
    print(render_body(json.loads(src.read_text()), event, dst))
