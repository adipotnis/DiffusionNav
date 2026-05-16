import os, pickle, numpy as np
root = "/work/nvme/bgbm/apotnis2/vjepa_nav_train/visualnav-transformer/train/datasets/recon"
out = []
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "traj_data.pkl")
    if not os.path.exists(p): continue
    n_imgs = sum(1 for f in os.listdir(os.path.join(root, d)) if f.endswith(".jpg"))
    try:
        with open(p, "rb") as f: dat = pickle.load(f)
        pos = np.asarray(dat["position"])
        N = min(len(pos), n_imgs)
        if N < 80: continue
        path_len = float(np.linalg.norm(np.diff(pos[:N], axis=0), axis=1).sum())
        win = 50
        max_disp = 0.0
        best_t = 0
        for i in range(0, N - win, 10):
            disp = float(np.linalg.norm(pos[i + win] - pos[i]))
            if disp > max_disp:
                max_disp = disp
                best_t = i
        out.append((d, N, path_len, max_disp, best_t))
    except Exception:
        pass
out.sort(key=lambda x: -x[3])
print(f"{'traj':70s} {'N':>5s} {'pathL':>8s} {'maxd50':>8s} {'startT':>6s}")
for r in out[:30]:
    print(f"{r[0]:70s} {r[1]:5d} {r[2]:8.2f} {r[3]:8.2f} {r[4]:6d}")
