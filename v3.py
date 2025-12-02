import mdtraj as mda
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

data_path = 'F:/university/UCUG1702/water 1/dump-surface.lammpstrj'
data_path = '/example_data.lammpstrj'
# data_path = 'F:/university/UCUG1702/water 1/dump-surface-280.lammpstrj'
# data_path = 'F:/university/UCUG1702/water 1/dump-surface-320.lammpstrj'

# 沿 z 轴方向将体系分层（用于分析不同 z 高度的水分子取向）
layer_num = 40

# 存储盒子边界的最大/最小坐标（后续从轨迹文件中读取）
max_pos = None  # np.ndarray shape=(3,)
min_pos = None  # np.ndarray shape=(3,)


class H2O:
    """水分子类，用于存储单个水分子的氧原子和两个氢原子坐标，并处理周期性边界问题"""

    def __init__(self, pO: np.ndarray, pH1: np.ndarray, pH2: np.ndarray) -> None:
        """
        初始化水分子坐标

        参数:
            pO: 氧原子坐标 (x,y,z)
            pH1: 第一个氢原子坐标 (x,y,z)
            pH2: 第二个氢原子坐标 (x,y,z)
        """
        global max_pos, min_pos

        self.pO = pO.copy()   # 氧原子坐标
        self.pH1 = pH1.copy() # 氢原子1坐标
        self.pH2 = pH2.copy() # 氢原子2坐标

        # 在 box 信息已知时做 PBC 修正
        if max_pos is not None and min_pos is not None:
            self.pH1 = self._fix_pH(self.pH1)
            self.pH2 = self._fix_pH(self.pH2)

    def _fix_pH(self, pH: np.ndarray) -> np.ndarray:
        """
        修正氢原子坐标以符合周期性边界条件

        思路：如果氢原子与氧原子的差向量超过 box_length / 2，
        就通过 ±box_length 把 H 拉回到与 O 最近的镜像。

        参数:
            pH: 氢原子当前坐标

        返回:
            修正后的氢原子坐标
        """
        global max_pos, min_pos

        p = pH.copy()
        for i in range(3):
            box = max_pos[i] - min_pos[i]
            if box == 0:
                continue

            delta = p[i] - self.pO[i]
            # 映射到 (-box/2, box/2]
            while delta > box / 2:
                delta -= box
                p[i] -= box
            while delta < -box / 2:
                delta += box
                p[i] += box

        return p


def calc_1_frame(data_raw: list[str], h2o_num: int, data_start_line: int) -> list[H2O]:
    """
    解析单个时间帧的数据，将原子坐标组装成水分子对象列表

    参数:
        data_raw: 当前帧的所有行（header + 原子行）
        h2o_num: 水分子数量（每个水分子包括3个原子）
        data_start_line: 当前帧在 data_raw 中的起始行号（一般为0）

    返回:
        当前帧所有水分子的列表（每个元素为 H2O 对象）
    """
    global min_pos, max_pos

    # 正确的 box 边界位置：TIMESTEP 开头算起的第 5, 6, 7 行
    box_bounds = data_raw[data_start_line + 5: data_start_line + 8]
    lb = np.array([float(s) for s in box_bounds[0].split()])
    rb = np.array([float(s) for s in box_bounds[2].split()])

    # 更新全局盒子最小/最大边界
    if min_pos is None:
        min_pos = lb.copy()
    else:
        min_pos = np.minimum(min_pos, lb)

    if max_pos is None:
        max_pos = rb.copy()
    else:
        max_pos = np.maximum(max_pos, rb)

    # 原子坐标起始行：头 9 行之后
    line_id = data_start_line + 9
    data: list[H2O] = []

    def get_pos(line: str, expected_type: int) -> np.ndarray:
        parts = line.strip().split()
        atom_type = int(parts[1])
        assert atom_type == expected_type, f"Expected type {expected_type}, got {atom_type}"
        return np.array([float(parts[2]), float(parts[3]), float(parts[4])])

    for _ in range(h2o_num):
        pO = get_pos(data_raw[line_id], 1)          # 氧原子（类型1）
        pH1 = get_pos(data_raw[line_id + 1], 2)     # 氢原子1（类型2）
        pH2 = get_pos(data_raw[line_id + 2], 2)     # 氢原子2（类型2）
        data.append(H2O(pO, pH1, pH2))              # 加入当前帧
        line_id += 3                                # 下一个水分子

    return data


# ================== 读取所有时间帧 ==================

frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

with open(data_path, "r") as f:
    total_line_count = sum(1 for _ in f)

process = tqdm(total=total_line_count, desc="Reading trajectory")

with open(data_path, "r") as f:
    while True:
        # 读取一帧的头信息（9 行）
        head = []
        for _ in range(9):
            line = f.readline()
            process.update()
            if not line:
                head = []
                break
            head.append(line)

        if not head:
            process.close()
            break

        assert "ITEM: TIMESTEP" in head[0]

        natoms = int(head[3])     # 这一帧原子总数
        h2o_num = natoms // 3     # 水分子数

        # 精确逐行读取 natoms 行
        atom_lines = []
        for _ in range(natoms):
            line = f.readline()
            if not line:
                break
            atom_lines.append(line)
            process.update()

        if len(atom_lines) < natoms:
            # 文件被截断或 test 文件不完整，直接结束
            process.close()
            break

        frame_data = calc_1_frame(head + atom_lines, h2o_num, 0)
        frames.append((
            np.array([h2o.pO for h2o in frame_data]),
            np.array([h2o.pH1 for h2o in frame_data]),
            np.array([h2o.pH2 for h2o in frame_data]),
        ))

process.close()

# 确认所有帧的水分子数一致
assert max(f[0].shape[0] for f in frames) == min(f[0].shape[0] for f in frames)

print(f"Total frames parsed: {len(frames)} with {frames[0][0].shape[0]} H2O each.")
print("box bounds:", min_pos, max_pos)

# ================== 按帧分层 ==================

layers: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(layer_num)]

for frame_idx, (pO, pH1, pH2) in enumerate(frames):
    frame_min_z = np.min(pO[:, 2])
    frame_max_z = np.max(pO[:, 2])
    dz = (frame_max_z - frame_min_z) / layer_num if layer_num > 0 else 0.0

    if dz == 0:
        mid_layer = layer_num // 2
        for layer_idx in range(layer_num):
            if layer_idx == mid_layer:
                layers[layer_idx].append((pO.copy(), pH1.copy(), pH2.copy()))
            else:
                layers[layer_idx].append(
                    (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
                )
        continue

    for layer_idx in range(layer_num):
        low = frame_min_z + layer_idx * dz
        if layer_idx < layer_num - 1:
            high = frame_min_z + (layer_idx + 1) * dz
            mask = (pO[:, 2] >= low) & (pO[:, 2] < high)
        else:
            high = frame_max_z
            mask = (pO[:, 2] >= low) & (pO[:, 2] <= high)

        layers[layer_idx].append((pO[mask], pH1[mask], pH2[mask]))


def compute_dipole_angles_for_layer(pO_layer: np.ndarray, 
                                    pH1_layer: np.ndarray, 
                                    pH2_layer: np.ndarray) -> np.ndarray:
    """
    计算一层中所有水分子的偶极矩与z轴（平面法向）的夹角（单位：度）
    
    偶极矩定义：从两个氢原子电荷中心指向氧原子的平均位置，向量为 (2O - (H1 + H2))
    夹角范围：0~180度（区分偶极矩向上/向下）
    
    参数:
        pO_layer: 该层所有氧原子坐标，shape=(N,3)
        pH1_layer: 该层所有氢原子1坐标，shape=(N,3)
        pH2_layer: 该层所有氢原子2坐标，shape=(N,3)
    
    返回:
        夹角数组，shape=(N,)
    """
    if pO_layer.shape[0] == 0:  # 若该层无水分子，返回空数组
        return np.array([])

    # 偶极矩方向向量（这里不关心绝对大小，只要方向）
    dipole = 2 * pO_layer - (pH1_layer + pH2_layer)  # shape=(N,3)
    
    # z轴方向（平面法向）
    z_axis = np.array([0, 0, 1])
    
    # 计算每个偶极矩与z轴的夹角
    z = dipole[:, 2]  # 偶极矩在z方向上的分量
    len_dipole = np.linalg.norm(dipole, axis=1)  # 偶极矩长度
    cos_angle = z / len_dipole  # 夹角的余弦
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数值稳定性处理，防止超出[-1,1]
    
    angle = np.arccos(cos_angle)  # 得到弧度
    return angle / np.pi * 180  # 转换成度


def vec_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两向量间夹角，返回角度（单位：deg）
    
    参数:
        vec1, vec2: 输入的三个维坐标向量
    
    返回:
        两向量之间夹角（0~180度）
    """
    d_prod = (vec1 * vec2).sum()  # 点积
    angle = np.arccos(d_prod / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))  # 夹角弧度
    return angle / np.pi * 180  # 转为度


def compute_tilt_angles_for_layer(pO_layer: np.ndarray, 
                                  pH1_layer: np.ndarray, 
                                  pH2_layer: np.ndarray) -> np.ndarray:
    """
    计算一层中所有水分子的倾斜角（tilt angle），及其与xy平面的夹角
    
    定义：
        - 偶极矩d是从两个氢原子电荷中心指向氧原子的位置向量：d = r_O - (r_H1 + r_H2)/2
        - 倾斜角θ：偶极矩与z轴的夹角

    参数:
        pO_layer: 氧原子坐标，shape=(N,3)
        pH1_layer: 氢原子1坐标，shape=(N,3)
        pH2_layer: 氢原子2坐标，shape=(N,3)
    
    返回:
        倾斜角数组（单位：度），shape=(N,)
    """
    N = pO_layer.shape[0]
    if N == 0:  # 如果当前层没有水分子
        return np.array([])

    # 电荷中心（氢原子的平均位置）
    h_center = (pH1_layer + pH2_layer) / 2.0  # shape=(N,3)
    # 偶极矩向量
    dipole = pO_layer - h_center  # shape=(N,3)

    # z方向单位向量
    z = np.array([0, 0, 1])

    # 每个偶极矩与z轴的夹角
    cos_theta = dipole @ z / np.linalg.norm(dipole, axis=1)  # shape=(N,)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
    theta = np.arccos(cos_theta)  # 弧度

    return theta / np.pi * 180  # 角度


def compute_plane_angles_for_layer(pO_layer: np.ndarray, 
                                   pH1_layer: np.ndarray, 
                                   pH2_layer: np.ndarray) -> np.ndarray:
    """
    计算一层中每个水分子平面与z轴的夹角（单位：度）
    
    定义：
        - 水分子平面由 (O, H1, H2) 三点构成
        - 法向量 n = (H1 - O) x (H2 - O)
        - 平面法向量与z轴之间的夹角
    
    参数:
        pO_layer: 氧原子坐标，shape=(N,3)
        pH1_layer: 氢原子1坐标，shape=(N,3)
        pH2_layer: 氢原子2坐标，shape=(N,3)
    
    返回:
        夹角数组（单位：度），shape=(N,)
    """
    N = pO_layer.shape[0]
    if N == 0:
        return np.array([])

    # 构造两个向量：OH1 和 OH2
    OH1 = pH1_layer - pO_layer
    OH2 = pH2_layer - pO_layer

    # 每个水分子平面的法向量 n = OH1 x OH2
    normals = np.cross(OH1, OH2)  # shape=(N,3)

    z = np.array([0, 0, 1])  # z轴方向
    # 计算每个法向量与z轴的夹角
    # angle = arccos( n_z / |n| )
    norm_normals = np.linalg.norm(normals, axis=1)
    # 去除法向量长度为0的情况（理论上不应该发生）
    valid_mask = norm_normals > 0
    normals = normals[valid_mask]
    norm_normals = norm_normals[valid_mask]

    cos_angle = normals[:, 2] / norm_normals
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数值稳定性
    angle = np.arccos(cos_angle)

    return angle / np.pi * 180  # 转为度


# 针对每一层计算所有帧的倾斜角分布
angle_dipole : list[list[np.ndarray]] = [[] for _ in range(layer_num)]  # 偶极矩角度
angle_tilt   : list[list[np.ndarray]] = [[] for _ in range(layer_num)]  # 倾斜角
angle_plane  : list[list[np.ndarray]] = [[] for _ in range(layer_num)]  # 平面法向角

# 遍历每一层、每一帧，计算各种角度
for layer_idx in range(layer_num):
    for frame_idx in tqdm(range(len(frames)), desc=f"Processing layer {layer_idx}/{layer_num}"):
        pO_layer, pH1_layer, pH2_layer = layers[layer_idx][frame_idx]
        # 如果该层该帧没有水分子，则跳过
        if pO_layer.shape[0] == 0:
            angle_dipole[layer_idx].append(np.array([]))
            angle_tilt[layer_idx].append(np.array([]))
            angle_plane[layer_idx].append(np.array([]))
            continue

        # 偶极矩与z轴的夹角
        angle_dipole[layer_idx].append(
            compute_dipole_angles_for_layer(pO_layer, pH1_layer, pH2_layer)
        )

        # 倾斜角
        angle_tilt[layer_idx].append(
            compute_tilt_angles_for_layer(pO_layer, pH1_layer, pH2_layer)
        )

        # 水分子平面法向与z轴的夹角
        angle_plane[layer_idx].append(
            compute_plane_angles_for_layer(pO_layer, pH1_layer, pH2_layer)
        )

# 统计并绘制每一层的各类角度分布（示例：偶极矩角度分布）

theta_bins = np.linspace(0, 180, 181)  # 0~180度按1度划分
theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2

# 对每层计算角度分布（直方图归一化为概率分布）
dipole_hist_per_layer = []
tilt_hist_per_layer   = []
plane_hist_per_layer  = []

for layer_idx in range(layer_num):
    # 将该层所有帧的角度数据拼接
    all_dipole_angles = np.concatenate(angle_dipole[layer_idx]) if len(angle_dipole[layer_idx]) else np.array([])
    all_tilt_angles   = np.concatenate(angle_tilt[layer_idx])   if len(angle_tilt[layer_idx]) else np.array([])
    all_plane_angles  = np.concatenate(angle_plane[layer_idx])  if len(angle_plane[layer_idx]) else np.array([])

    # 如果该层完全没有数据，则记录空
    if all_dipole_angles.size == 0:
        dipole_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_dipole_angles, bins=theta_bins, density=True)
        dipole_hist_per_layer.append(hist)

    if all_tilt_angles.size == 0:
        tilt_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_tilt_angles, bins=theta_bins, density=True)
        tilt_hist_per_layer.append(hist)

    if all_plane_angles.size == 0:
        plane_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_plane_angles, bins=theta_bins, density=True)
        plane_hist_per_layer.append(hist)

# 示例：画出所有层的偶极角分布随角度变化的图（可以选若干层展示）
plt.figure(figsize=(8, 6))
for layer_idx in range(layer_num):
    prob = dipole_hist_per_layer[layer_idx]
    if np.all(prob == 0):  # 完全无数据的层跳过
        continue
    z_mid = min_pos[2] + (max_pos[2] - min_pos[2]) * (layer_idx + 0.5) / layer_num
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}, z~{z_mid:.2f}')

plt.xlabel(r'$\theta_{\mu-z}$ (deg)')
plt.ylabel('Probability')
plt.legend()
plt.title('Dipole angle distribution per layer')
plt.tight_layout()
plt.savefig("dipole_angle_per_layer.png", dpi=300)
plt.show()


# 可选：针对某一层画详细分布图（例如第20层）
target_layer = 20
if 0 <= target_layer < layer_num:
    plt.figure(figsize=(8, 6))
    prob = dipole_hist_per_layer[target_layer]
    plt.bar(theta_mid, prob, width=1.0)
    plt.xlabel(r'$\theta_{\mu-z}$ (deg)')
    plt.ylabel('Probability')
    plt.title(f'Dipole angle distribution in layer {target_layer}')
    plt.tight_layout()
    plt.savefig(f"dipole_angle_layer_{target_layer}.png", dpi=300)
    plt.show()


# 示例：也可以对倾斜角、平面角做类似的分析和绘图
# 这里给出一个简单示例，对倾斜角在若干层进行比较

selected_layers = [5, 10, 15, 20, 25, 30]  # 选取的一些层
plt.figure(figsize=(8, 6))
for layer_idx in selected_layers:
    if layer_idx >= layer_num:
        continue
    prob = tilt_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}')

plt.xlabel(r'$\theta_{\mathrm{tilt}}$ (deg)')
plt.ylabel('Probability')
plt.title('Tilt angle distribution (selected layers)')
plt.legend()
plt.tight_layout()
plt.savefig("tilt_angle_selected_layers.png", dpi=300)
plt.show()

# 对平面法向角进行简单绘制（示例）
plt.figure(figsize=(8, 6))
for layer_idx in selected_layers:
    if layer_idx >= layer_num:
        continue
    prob = plane_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}')

plt.xlabel(r'$\theta_{\mathrm{plane-z}}$ (deg)')
plt.ylabel('Probability')
plt.title('Plane normal angle distribution (selected layers)')
plt.legend()
plt.tight_layout()
plt.savefig("plane_angle_selected_layers.png", dpi=300)
plt.show()


# 若想分别输出每一层（或某几层）的角度分布数据，也可以保存为文件，便于后续处理
# 示例: 保存某层的偶极角概率分布为txt
save_layer = 10
if 0 <= save_layer < layer_num:
    np.savetxt(f"dipole_angle_distribution_layer_{save_layer}.txt",
               np.vstack([theta_mid, dipole_hist_per_layer[save_layer]]).T,
               header="theta(deg) probability")

# 多层偶极分布对比（示例）
plt.figure(figsize=(10, 6))
for layer_idx in range(0, layer_num, 5):  # 每隔5层画一次，避免太乱
    prob = dipole_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}')
plt.xlabel(r'$\theta_{\mu-n}$ (deg)', fontsize=8)
plt.ylabel('Probability', fontsize=8)  # 概率
plt.xlim(0, 180)  # 角度范围固定为0-180度
# 自适应Y轴：最大值×1.1，留余量
prob_max = np.max(prob)
plt.ylim(0, prob_max * 1.1)
# 优化刻度：自动生成合理刻度（可选，增强可读性）
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # 最多显示5个Y轴刻度
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("dipole_angle_per_layer_multi.png", dpi=300)
plt.show()


# 针对每层分别绘制偶极矩角度分布（每层一张子图，可视化层间差异）
fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(15, 20), sharex=True, sharey=False)
axes = axes.flatten()

for layer_idx in range(layer_num):
    ax = axes[layer_idx]
    prob = dipole_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    ax.plot(theta_mid, prob)
    ax.set_title(f'Layer {layer_idx}', fontsize=8)
    ax.set_xlabel(r'$\theta_{\mu-n}$ (deg)', fontsize=8)
    ax.set_ylabel('Probability', fontsize=8)  # 概率
    ax.set_xlim(0, 180)  # 角度范围固定为0-180度
    # 自适应Y轴：最大值×1.1，留余量
    prob_max = np.max(prob)
    ax.set_ylim(0, prob_max * 1.1)
    # 优化刻度：自动生成合理刻度（可选，增强可读性）
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # 最多显示5个Y轴刻度
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# 调整子图间距
plt.tight_layout()
plt.savefig("dipole_angle_per_layer.png", dpi=300)
plt.show()
