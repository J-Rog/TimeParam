"""Camera contract shared by collection, training and the deployed WoR agent."""
import numpy as np


def camera_specs(config, sensor_tick=0.05):
    common = dict(type="sensor.camera.rgb", x=config["camera_x"], y=0,
                  z=config["camera_z"], roll=0.0, pitch=0.0, height=240,
                  sensor_tick=sensor_tick)
    return [dict(common, yaw=yaw, width=160, fov=60, id=f"Wide_RGB_{i}")
            for i, yaw in enumerate((-55.0, 0.0, 55.0))] + [
        dict(common, yaw=0.0, width=384, fov=50, id="Narrow_RGB")]


def stitch_rgb(wide_bgra, narrow_bgra):
    """CARLA BGRA -> uncropped RGB panorama (left, center, right) and narrow."""
    wide = np.concatenate([im[..., :3][..., ::-1] for im in wide_bgra], axis=1)
    return wide.copy(), narrow_bgra[..., :3][..., ::-1].copy()


def crop_rgb(wide, narrow, config):
    top, bottom = int(config["wide_crop_top"]), int(config["narr_crop_bottom"])
    if not 0 <= top < wide.shape[0] or not 0 <= bottom < narrow.shape[0]:
        raise ValueError("Camera crop would produce an empty image")
    return wide[top:].copy(), narrow[:narrow.shape[0] - bottom].copy()
