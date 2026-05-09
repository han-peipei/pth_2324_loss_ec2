
# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from train_3_B import train_and_evaluate_from_npy
from scipy.spatial import cKDTree
import datetime
from datetime import timedelta
from datetime import datetime

# ---------------- 必要的小工具函数（保留） ----------------
def to_array(x):
    if isinstance(x, list):
        try:
            return np.stack([np.asarray(e) for e in x], axis=0)
        except Exception:
            return np.asarray(x)
    return np.asarray(x)

def interpolate_to_station(coords, tree, nwp_data):
    """
    使用最邻近插值法将格点数据插值到站点经纬度
    """
    dist, idx = tree.query(coords)  # 查找每个站点最近的格点
    return nwp_data[:, idx]  # 获取对应格点数据

# ---------------- 配置区 ----------------
# PY_DIR = os.path.dirname(os.path.abspath(__file__))
# JOB_DIR = os.environ.get("JOB_DIR", PY_DIR)             # bash 所在目录（推荐从 bash 传入）
# BASE_DIR = os.path.abspath(os.path.join(JOB_DIR, "..")) # bash 的上一级目录
# station_id=[54456]
# time=(datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d%H")
# ROOT  = os.path.join(BASE_DIR, "download_48h","train", f"{station_id[0]}_{time[:-2]}",str(station_id[0]))

# ROOT = "/mnt/g/test_24_train/54456"   # 
ROOT = "/kaggle/input/datasets/niaosilius/stations-25-train-drop-08/stations_25_train_drop_08"   # 
# csv_path =os.path.join(BASE_DIR, "download_48h", "station",f"{station_id[0]}_{time[:-2]+'06'}_past48h.csv")
csv_path = "2023.csv"  # ← 站点信息CSV
VARS = ("10u", "10v")                                         # 使用的变量
# VARS = ("10u", "10v",'2DPT','2RH','CAPE','MSL','2T','VIS')                                         # 使用的变量
TI = "08"                                                     # 起报时（当前正则里只匹配02）
_var_pat = "|".join(map(re.escape, VARS))
_station_pat = r'(?:[A-Za-z]\d{4}|\d{5})'
_lat_re = re.compile(rf'^lat_({_station_pat})\.npy$')
_lon_re = re.compile(rf'^lon_({_station_pat})\.npy$')
# _station_pat = r'(?:[A]\d{4})'
# _station_pat =['A2662','A3171']
_data_re  = re.compile(rf'^train_data_({_var_pat})_({_station_pat})_{TI}\.npy$')
_obs_re = re.compile(rf'^train_labels_({_station_pat})_{TI}\.npy$')
_time_re= re.compile(rf'^train_time_({_station_pat})_{TI}\.npy$')  # 可选

data_idx,obs_idx, time_idx,lat_idx,lon_idx = {}, {}, {}, {} ,{} 
stations_set = set()

for fn in os.listdir(ROOT):
    m = _data_re.match(fn)
    if m:
        var, station = m.groups()
        data_idx[("train", station, var)] = os.path.join(ROOT, fn)
        stations_set.add(station)
        continue
    m = _obs_re.match(fn)
    if m:
        (station,) = m.groups()
        obs_idx[("train", station)] = os.path.join(ROOT, fn)
        stations_set.add(station)
        continue
    m = _lat_re.match(fn)
    if m:
        station = m.group(1)
        lat_idx[station] = os.path.join(ROOT, fn)
        stations_set.add(station)
        continue

    m = _lon_re.match(fn)
    if m:
        station = m.group(1)
        lon_idx[station] = os.path.join(ROOT, fn)
        stations_set.add(station)
        continue

    m = _time_re.match(fn)
    if m:
        station = m.group(1)
        time_idx[station] = os.path.join(ROOT, fn)
        stations_set.add(station)
        continue
# print(time_idx)
# print(time_idx[station])
# time=np.load(time_idx[station],allow_pickle=True)
# print(time)
stations = sorted(stations_set)
if not stations:
    raise RuntimeError("未在目录中匹配到任何站点文件，请检查ROOT与命名规则/起报时。")

# ---------------- 读取站点经纬高程并生成 coords ----------------
df = pd.read_csv(csv_path, dtype={'Station_Id_C': str}, low_memory=False)
df['Station_Id_C'] = df['Station_Id_C'].str.strip().str.upper()

dfu = (df.drop_duplicates('Station_Id_C', keep='last')
         .set_index('Station_Id_C')[['Lat','Lon','Alti']])

# 构建查找表
station_lut = {sid: (float(row['Lat']), float(row['Lon']), float(row['Alti']))
               for sid, row in dfu.iterrows()}
# ---------------- 逐站加载并拼接（train/val） ----------------
hist_train, nwp_train, y_train = [], [], []
spd_train=[]
# hist_val,   nwp_val,   y_val   = [], [], []
meta_tr = []
time_train=[]
# meta_tr, meta_va = [], []  

Hhist = 24   # 历史窗口长度（上一个24小时）
F     = 24   # 预测窗口长度（下一个24小时）
S     = 1 
# hist_list  = []
# nwp_list   = []
# spd_list   = []
# coords_list = []
# time_future_list = []
# ytrue_list = []
# meta_station = []   # 与 batch 对应的站点顺序
# def _to_1d_float(x):
#     x = np.asarray(x)
#     return x.reshape(-1).astype(np.float32)

# def pick_tsplit(Tlen, Hhist, F):
#     """
#     默认选择最后一个可用的“未来24小时”窗口：
#     """
#     t_split = Tlen - F
#     if t_split < Hhist:
#         return None
#     return t_split

# for split in ("train",):
for station in stations:
    # 检查变量与标签是否齐全
    missing = [v for v in VARS if ("train", station, v) not in data_idx]
    if missing or ("train", station) not in obs_idx:
        continue
    time_1 = np.load(time_idx[station], allow_pickle=True)   # 1D: (T,)
    try:
        var_arrs = []
        # Nwin = None
        for v in VARS:
            modle = np.load(data_idx[("train", station, v)], allow_pickle=True)
            print("[DBG]", station, v, modle.shape)
            Tlen, H, W = modle.shape
            modle = modle.reshape(Tlen, 1, H, W)
            var_arrs.append(modle)

        X_full = np.concatenate(var_arrs, axis=1)
        # 3) 读取标签，整理为 (T,)
        lab_np=np.load(obs_idx[("train", station)], allow_pickle=True)
        y_full = lab_np.reshape(-1) if lab_np.ndim > 1 else lab_np
        Tlen = X_full.shape[0]
        need = Hhist + F
        nwin = (Tlen - need) // S + 1

        H_list, N_list, Y_list , spd_list,time_list= [], [], [],[],[]
##############################################################################################################################################
        ij = np.load(os.path.join(ROOT, f"nearest_ij_{station}.npy")).astype(int)
        iy, ix = int(ij[0]), int(ij[1])
        u_sta = X_full[:, 0, iy, ix]     # (F,)
        v_sta = X_full[:, 1, iy, ix]     # (F,)
        spd_sta = np.sqrt(u_sta*u_sta + v_sta*v_sta)   # (F,)
        
##############################################################################################################################################
        for k in range(nwin):
            t0 = k * S
            t1 = t0 + Hhist
            t2 = t1 + F
        # t0 = 0
        # t1 = t0 + Hhist
        # t2 = t1 + F
            hist = y_full[t0:t1]  
            nwp  = X_full[t1:t2]  
            y    = y_full[t1:t2] 
            spd  =spd_sta[t1:t2]
            time_2=time_1[t1:t2]
###############################################################################################################################################
###############################################################################################################################################
 
            H_list.append(hist)
            N_list.append(nwp)
            Y_list.append(y)
            spd_list.append(spd)
            time_list.append(time_2)

        
        H_win = np.stack(H_list, axis=0)  # (nwin, Hhist)
        N_win = np.stack(N_list, axis=0)  # (nwin, F, C, H, W)
        Y_win = np.stack(Y_list, axis=0)  # (nwin, F)
        spd_win = np.stack(spd_list, axis=0)  # (nwin, F)
        time_win = np.stack(time_1, axis=0)  # (nwin, F)

        hist_train.append(H_win)
        nwp_train.append(N_win)
        y_train.append(Y_win)
        spd_train.append(spd_win)
        time_train.append(time_win)

        meta_tr.append((station, nwin)) 

        # print(f"[OK][{station}] hist{hist.shape} nwp{nwp.shape} spd{spd.shape} t_split={t_split}")

    except Exception as e:
        print(f"[FAIL][{station}] {e}")


hist_train_all   = np.concatenate(hist_train, axis=0) if hist_train else None
model_train_all  = np.concatenate(nwp_train,   axis=0) if nwp_train   else None
obs_train_all    = np.concatenate(y_train,  axis=0) if y_train  else None
spd_train_all    = np.concatenate(spd_train,  axis=0) if spd_train  else None
time    = np.concatenate(time_train,  axis=0) if time_train  else None

print("hist_train_all:", None if hist_train_all is None else hist_train_all.shape)
print("model_train_all:", None if model_train_all is None else model_train_all.shape)



def coords_from_meta(meta_list, lut):
    blocks = []
    for sid, nwin in meta_list:
        key = str(sid).strip().upper()
        if key not in lut:
            raise KeyError(f"站点 {key} 不在站点表里")
        lat, lon, elev = lut[key]
        blocks.append(np.repeat([[lat, lon, elev]], nwin, axis=0).astype(np.float32))
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0,3), np.float32)

coords_tr = coords_from_meta(meta_tr, station_lut)   # (sum(nwin_train), 3)
# coords_va = coords_from_meta(meta_va, station_lut)   # (sum(nwin_val), 3)

print("\n========== 开始统一训练模型 ==========\n")

# 你的训练函数应接受四个输入（不需要 hour_codes）
model = train_and_evaluate_from_npy(
    hist_train_all, model_train_all,   obs_train_all,
    # hist_val_all,model_val_all,obs_val_all,
    coords_tr,  
    spd_train_all,
    time,
    # coords_va,
    device=torch.device('cuda')
)
