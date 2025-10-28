import numpy as np
from numba import njit, prange

# =========================
# Low-level helpers (Numba)
# =========================

@njit(fastmath=True)
def _ray_aabb(o, d, bmin, bmax, max_range):
    invdx = 1.0 / (d[0] if np.abs(d[0]) > 1e-12 else 1e-12)
    invdy = 1.0 / (d[1] if np.abs(d[1]) > 1e-12 else 1e-12)
    invdz = 1.0 / (d[2] if np.abs(d[2]) > 1e-12 else 1e-12)

    t0x = (bmin[0] - o[0]) * invdx; t1x = (bmax[0] - o[0]) * invdx
    t0y = (bmin[1] - o[1]) * invdy; t1y = (bmax[1] - o[1]) * invdy
    t0z = (bmin[2] - o[2]) * invdz; t1z = (bmax[2] - o[2]) * invdz

    tmin = max(min(t0x, t1x), min(t0y, t1y), min(t0z, t1z))
    tmax = min(max(t0x, t1x), max(t0y, t1y), max(t0z, t1z))

    if tmax < max(tmin, 0.0):
        return False, 0.0, 0.0

    t_enter = tmin if tmin > 0.0 else 0.0
    if max_range > 0.0 and t_enter > max_range:
        return False, 0.0, 0.0
    if max_range > 0.0 and tmax > max_range:
        tmax = max_range
    return True, t_enter, tmax


@njit(fastmath=True)
def _axis_setup(i, x0, dx, N, vsize, org):
    """
    Compute tMax and tDelta for one axis in 3D DDA.
    """
    if dx > 0.0:
        next_boundary = org + (i + 1) * vsize
    else:
        next_boundary = org + (i) * vsize

    if np.abs(dx) < 1e-12:
        tMax = 1e30
        tDelta = 1e30
    else:
        tMax = (next_boundary - x0) / dx
        tDelta = vsize / np.abs(dx)
    return tMax, tDelta


# =========================
# Core kernel (Numba, DDA)
# =========================

@njit(parallel=True, fastmath=True)
def visible_mask_from_camera_numba(
    occ,                  # bool [X,Y,Z]
    ox, oy, oz,           # float64
    vx, vy, vz,           # float64
    Kinv,                 # float64(3,3)
    R_world,              # float64(3,3)
    cam_o,                # float64(3,)
    Wimg, Himg,           # int
    stride,               # int
    max_range             # float
):
    X, Y, Z = occ.shape
    grid_min = np.array([ox, oy, oz], dtype=np.float64)
    grid_max = np.array([ox + vx*X, oy + vy*Y, oz + vz*Z], dtype=np.float64)

    visible = np.zeros((X, Y, Z), dtype=np.uint8)

    # parfor: step은 1로 고정, stride는 내부에서 modulo로 건너뜀
    for v in prange(Himg):
        if stride > 1 and (v % stride) != 0:
            continue
        for u in range(Wimg):
            if stride > 1 and (u % stride) != 0:
                continue

            # d_cam = Kinv @ [u, v, 1]
            d_cam0 = Kinv[0,0]*u + Kinv[0,1]*v + Kinv[0,2]
            d_cam1 = Kinv[1,0]*u + Kinv[1,1]*v + Kinv[1,2]
            d_cam2 = Kinv[2,0]*u + Kinv[2,1]*v + Kinv[2,2]

            # world dir = R_world @ d_cam
            dw0 = R_world[0,0]*d_cam0 + R_world[0,1]*d_cam1 + R_world[0,2]*d_cam2
            dw1 = R_world[1,0]*d_cam0 + R_world[1,1]*d_cam1 + R_world[1,2]*d_cam2
            dw2 = R_world[2,0]*d_cam0 + R_world[2,1]*d_cam1 + R_world[2,2]*d_cam2

            invlen = 1.0 / (np.sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2) + 1e-12)
            dw0 *= invlen; dw1 *= invlen; dw2 *= invlen

            # ray-box
            hit, t_enter, t_exit = _ray_aabb(
                cam_o, np.array([dw0,dw1,dw2]), grid_min, grid_max, max_range
            )
            if not hit:
                continue

            eps = 1e-6
            x0 = cam_o[0] + dw0 * (t_enter + eps)
            y0 = cam_o[1] + dw1 * (t_enter + eps)
            z0 = cam_o[2] + dw2 * (t_enter + eps)

            ix = int(np.floor((x0 - ox) / vx))
            iy = int(np.floor((y0 - oy) / vy))
            iz = int(np.floor((z0 - oz) / vz))
            if ix < 0 or iy < 0 or iz < 0 or ix >= X or iy >= Y or iz >= Z:
                continue

            sx = 1 if dw0 >= 0.0 else -1
            sy = 1 if dw1 >= 0.0 else -1
            sz = 1 if dw2 >= 0.0 else -1

            # tMax/tDelta (inline, 분기 최소화)
            if dw0 > 0.0:
                nbx = ox + (ix + 1) * vx
            else:
                nbx = ox + ix * vx
            if abs(dw0) < 1e-12:
                tMaxX = 1e30; tDeltaX = 1e30
            else:
                tMaxX = (nbx - x0) / dw0
                tDeltaX = vx / abs(dw0)

            if dw1 > 0.0:
                nby = oy + (iy + 1) * vy
            else:
                nby = oy + iy * vy
            if abs(dw1) < 1e-12:
                tMaxY = 1e30; tDeltaY = 1e30
            else:
                tMaxY = (nby - y0) / dw1
                tDeltaY = vy / abs(dw1)

            if dw2 > 0.0:
                nbz = oz + (iz + 1) * vz
            else:
                nbz = oz + iz * vz
            if abs(dw2) < 1e-12:
                tMaxZ = 1e30; tDeltaZ = 1e30
            else:
                tMaxZ = (nbz - z0) / dw2
                tDeltaZ = vz / abs(dw2)

            max_steps = X + Y + Z + 3
            steps = 0
            t_cur = t_enter

            while 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z:
                visible[ix, iy, iz] = 1
                if occ[ix, iy, iz]:
                    break

                if tMaxX <= tMaxY and tMaxX <= tMaxZ:
                    ix += sx; t_cur = tMaxX; tMaxX += tDeltaX
                elif tMaxY <= tMaxZ:
                    iy += sy; t_cur = tMaxY; tMaxY += tDeltaY
                else:
                    iz += sz; t_cur = tMaxZ; tMaxZ += tDeltaZ

                steps += 1
                if steps > max_steps or t_cur > t_exit:
                    break

    return visible.view(np.bool_)


# =========================
# Python wrapper utilities
# =========================

def build_visible_mask_numba(
    occ, origin, voxel_size, K, R, t, img_wh, stride=1, max_range=-1.0
):
    occ = np.asarray(occ, dtype=np.bool_)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64).reshape(3,3)
    R = np.asarray(R, dtype=np.float64).reshape(3,3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    W, H = int(img_wh[0]), int(img_wh[1])

    Kinv = np.linalg.inv(K).astype(np.float64)
    R_world = R.T.astype(np.float64)
    cam_o = (-R_world @ t).astype(np.float64)

    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    vx, vy, vz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])

    vis = visible_mask_from_camera_numba(
        occ, ox, oy, oz, vx, vy, vz, Kinv, R_world, cam_o,
        W, H, int(stride), float(max_range)
    )
    return vis


def build_visible_mask_multi_camera_numba(
    occ,
    origin,
    voxel_size,
    cameras,
    img_wh,
    stride=1,
    max_range=-1.0,
):
    """
    Multi-camera variant: OR-accumulate visible masks from several cameras.
    Args:
        occ, origin, voxel_size: same as single-camera.
        cameras: list of dicts with keys {'K','R','t'} per camera.
                 R,t are world->camera extrinsics.
        img_wh: (W, H)
        stride, max_range: see single-camera version.
    Returns:
        visible_all: np.bool_ [X, Y, Z]
    """
    visible_all = np.zeros_like(occ, dtype=np.bool_)
    for cam in cameras:
        vis = build_visible_mask_numba(
            occ, origin, voxel_size,
            cam['K'], cam['R'], cam['t'],
            img_wh, stride=stride, max_range=max_range
        )
        visible_all |= vis
    return visible_all


# =========================
# Minimal self-test
# =========================

if __name__ == "__main__":
    # Dummy grid
    X, Y, Z = 64, 64, 32
    occ = np.zeros((X, Y, Z), dtype=np.bool_)
    # Put an occupied wall around x=40 to create a first hit
    occ[40, 16:48, 8:24] = True

    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    voxel_size = np.array([0.1, 0.1, 0.1], dtype=np.float64)

    # Simple pinhole camera
    fx = fy = 500.0
    cx = 320.0
    cy = 240.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # Camera at the grid min corner, looking along +x (world)
    R = np.eye(3, dtype=np.float64)  # world->camera (identity means camera axes == world axes)
    t = np.zeros(3, dtype=np.float64)

    img_wh = (640, 480)

    # First call triggers JIT compilation
    vis = build_visible_mask_numba(
        occ, origin, voxel_size, K, R, t, img_wh, stride=2, max_range=-1.0
    )
    print("Visible mask:", vis.shape, vis.dtype, "num_true:", int(vis.sum()))
