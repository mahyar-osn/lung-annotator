import sys
import ctypes as ct

import cv2
import numpy as np


showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True


def on_mouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True


cv2.namedWindow("show3d")
cv2.moveWindow("show3d", 0, 0)
cv2.setMouseCallback("show3d", on_mouse)

dll = np.ctypeslib.load_library("utils/render", ".")


def show_points(xyz, c_gt=None, c_pred=None, wait_time=0,
                show_rot=False, magnify_blue=0, freeze_rot=False, background=(0, 0, 0),
                normalise_color=True, ball_radius=10):
    global showsz, mousex, mousey, zoom, changed
    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz
    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype="float32") + 255
        c1 = np.zeros((len(xyz),), dtype="float32") + 255
        c2 = np.zeros((len(xyz),), dtype="float32") + 255
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    if normalise_color:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    c0 = np.require(c0, "float32", "C")
    c1 = np.require(c1, "float32", "C")
    c2 = np.require(c2, "float32", "C")

    show = np.zeros((showsz, showsz, 3), dtype="uint8")

    def render():
        rotmat = np.eye(3)
        if not freeze_rot:
            xangle = (mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(xangle), -np.sin(xangle)],
                [0.0, np.sin(xangle), np.cos(xangle)],
            ]))
        if not freeze_rot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(
            np.array([
                [np.cos(yangle), 0.0, -np.sin(yangle)],
                [0.0, 1.0, 0.0],
                [np.sin(yangle), 0.0, np.cos(yangle)],
            ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype("int32")
        show[:] = background
        dll.render(
            ct.c_int(show.shape[0]), ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p), ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p), c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p), c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ball_radius))

        if magnify_blue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=0))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=1))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))
        if show_rot:
            cv2.putText(show, f"x angle {int(xangle / np.pi * 180)}",
                        (30, showsz - 30), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, f"y angle {int(yangle / np.pi * 180)}",
                        (30, showsz - 50), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, f"zoom {int(zoom * 100)}", (30, showsz - 70), 0,
                        0.5, cv2.cv.CV_RGB(255, 0, 0))

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow("show3d", show)
        if wait_time == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(wait_time) % 256
        if cmd == ord("q"):
            break
        elif cmd == ord("Q"):
            sys.exit(0)

        if cmd == ord("t") or cmd == ord("p"):
            if cmd == ord("t"):
                if c_gt is None:
                    c0 = np.zeros((len(xyz),), dtype="float32") + 255
                    c1 = np.zeros((len(xyz),), dtype="float32") + 255
                    c2 = np.zeros((len(xyz),), dtype="float32") + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c0 = np.zeros((len(xyz),), dtype="float32") + 255
                    c1 = np.zeros((len(xyz),), dtype="float32") + 255
                    c2 = np.zeros((len(xyz),), dtype="float32") + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalise_color:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            c0 = np.require(c0, "float32", "C")
            c1 = np.require(c1, "float32", "C")
            c2 = np.require(c2, "float32", "C")
            changed = True

        if cmd == ord("n"):
            zoom *= 1.1
            changed = True
        elif cmd == ord("m"):
            zoom /= 1.1
            changed = True
        elif cmd == ord("r"):
            zoom = 1.0
            changed = True
        elif cmd == ord("s"):
            cv2.imwrite("show3d.png", show)
        if wait_time != 0:
            break
    return cmd
