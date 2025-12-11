我直接先给你一份“期末开卷速查笔记大纲版”。
内容按**主题 + 周次**整理，方便你考试时用搜索（Ctrl+F）快速定位。

---

# UCUG1702 期末开卷速查总表

> 建议用法：
>
> * 先整体通读一遍抓逻辑
> * 考试时用关键词搜：比如“Fick”“Tg”“Beer Lambert”“应力应变”等
> * 所有公式尽量都配了：**中英文术语 + 字母含义**

---

## 0. 全局知识结构（从原子 → 能带 → 电/光/力/扩散）

1. **原子与键合**（Intro 1 & 2）

   * 原子结构、量子数、电子层、周期表、化学键类型
2. **晶体与缺陷**（Intro 3 & 4）

   * 晶体结构（FCC/BCC/HCP）、密排、密度计算、米勒指数、X 射线衍射
   * 点缺陷、位错、晶界；能带结构、金属/半导体/绝缘体
3. **电学性质**（Week5）

   * 电流、电流密度、电导率/电阻率、Drude 模型、自由电子/空穴、欧姆定律、超导、量子隧穿 
4. **光学性质**（Week6 – 重点）

   * 光的波粒二象性、反射/折射/吸收/透射、折射率、Beer-Lambert、带隙与颜色、介电常数、激发/发光、激光 
5. **力学性质 1（金属/一般固体）**（Week8 – 重点）

   * 应力-应变曲线、弹性/塑性、屈服强度、强度、韧性、延展性、刚度等
6. **力学性质 2（聚合物 + 相变）**（Week8/9 – 重点）

   * 聚合物玻璃态/橡胶态/半结晶、玻璃化转变温度 Tg、熔点 Tm、Tg/Tm 影响因素、半结晶双相结构 
7. **扩散**（Week10 – 重点）

   * Fick 第一定律与第二定律、扩散通量、扩散系数 Arrhenius 形式、误差函数解、Brownian motion、与电导/热导类比 

---

## 1. 原子结构与键合（Intro 1_atom, 2_bond）

### 1.1 原子结构 Atom structure

**关键术语**

* Proton 质子
* Neutron 中子
* Electron 电子
* Atomic number 原子序数：(Z)（核内质子数）
* Mass number 质量数：(A = Z + N)（N 为中子数）
* Isotope 同位素：A 不同但 Z 相同
* Bohr model 玻尔模型：电子在定态轨道上，能量量子化
* Quantum numbers 量子数：

  * (n) 主量子数（能级）
  * (l) 角量子数（0,1,2,…；s,p,d,f 轨道）
  * (m_l) 磁量子数
  * (m_s) 自旋量子数（±½）

**常见公式 & 字母释义**

* 氢原子能级（Bohr）：
  [
  E_n = -\frac{13.6\ \text{eV}}{n^2}
  ]

  * (E_n)：第 n 个能级能量
  * (n)：主量子数

* 波尔半径 Bohr radius：
  [
  a_0 \approx 0.529\ \text{Å}
  ]
  金属自由电子平均间距经常和 (a_0) 比较（如 Week5 里 rs/a0 ~ 2–6）。

**容易考的问题思路**

* 解释**原子能级为什么量子化**（边界条件 + 波动性）
* 画出简单能级图，说明从高能级跃迁到低能级会发光（光子能量 (E = h\nu)）

---

### 1.2 键合类型 & 势能曲线（Bonding）

**键类型（中英对照）**

* Ionic bond 离子键
* Covalent bond 共价键
* Metallic bond 金属键
* van der Waals bond 范德华键
* Hydrogen bond 氢键

**势能曲线**

* 一般形状：

  * 短程强排斥 + 中程吸引 → 形成势能井
  * 平衡键长处势能最低 = 结合能
* 函数示意（Lennard-Jones 势 L-J potential）：
  [
  U(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]
  ]

  * (U(r))：势能
  * (r)：原子间距
  * (\varepsilon)：势阱深度（结合强度）
  * (\sigma)：特征长度参数

**物理解读**

* 势阱越深 → 结合能越大 → 熔点、弹性模量等越高
* 势能曲线不对称 → 热膨胀（平均间距随 T 增大）

---

## 2. 晶体结构、X 射线和能带（Intro 3 & 4）

### 2.1 晶体结构（Crystal structure）

**晶体 & 非晶**

* Crystal 晶体：长程有序（long-range order）
* Amorphous 非晶：只有短程有序（short-range order）

**常见晶体结构**

* BCC 体心立方：Body-Centered Cubic

  * 每晶胞原子数：2
  * 配位数：8
  * 原子堆积因子 APF ≈ 0.68
* FCC 面心立方：Face-Centered Cubic

  * 每晶胞原子数：4
  * 配位数：12
  * APF ≈ 0.74（最致密金属之一）
* HCP 密排六方：Hexagonal Close-Packed

  * 配位数：12
  * APF ≈ 0.74

**重要符号**

* (a, b, c)：晶格常数（晶胞边长）
* Miller indices 米勒指数 ((hkl))：晶面指数
* Direction [uvw]：晶向

**立方晶体晶面间距**

[
d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}}
]

---

### 2.2 X 射线衍射（X-ray diffraction）

**Bragg 定律：**

[
n\lambda = 2 d \sin\theta
]

* (n)：衍射级数（整数）
* (\lambda)：X 射线波长
* (d)：晶面间距
* (\theta)：入射角（Bragg 角）

**思路**

* A 已知 λ 和 θ，求 d → 确定晶格常数 a
* B 已知结构和 a，求理论 d_{hkl}，与实验峰位对比确认晶体结构

---

### 2.3 缺陷（Defects）

**点缺陷（0D）**

* Vacancy 空位：晶格点上原子缺失
* Self-interstitial 自间隙：本征原子占据间隙位置
* Substitutional substitutional defect 置换缺陷：外来原子占据晶格点

**线缺陷（1D）**

* Dislocation 位错

  * Edge dislocation 刃型位错
  * Screw dislocation 螺型位错
* Burgers vector 伯格斯矢量 (\vec{b})：描述位错强度与方向
* 位错密度增加 → 强度提高（加工硬化）

**面缺陷（2D）**

* Grain boundary 晶界
* Twin boundary 孪晶界
* Stacking fault 堆垛层错

---

### 2.4 能带结构（Band structure）

**核心概念**

* Valence band 价带
* Conduction band 导带
* Band gap 带隙 (E_g)
* Fermi level 费米能级 (E_F)

**材料分类（按 Eg）**

* Metal 金属：价带与导带重叠或无带隙 → 高导电
* Semiconductor 半导体：Eg ~ 0.5–3 eV
* Insulator 绝缘体：Eg > 3–5 eV

**光学吸收与带隙（Week6 也用到）**

* 若光子能量 (h\nu > E_g) 则可产生跃迁 → 吸收光子
* 课件给出：

  * (E_g < 1.8\ \text{eV})：几乎吸收所有可见光 → 不透明 
  * (E_g > 3.1\ \text{eV})：对可见光几乎不吸收 → 透明无色（如金刚石）

---

## 3. 电学性质（Week5 – 概要）

### 3.1 电流与电流密度

**定义**

* Electric current 电流 (I)：单位时间通过某截面的净电荷量
* Current density 电流密度 (J)：单位面积的电流
  [
  J = \frac{I}{A}
  ]

**Drude 模型给出的微观表达式：** 

[
I = n e \bar v A,\quad J = n e \bar v
]

* (n)：自由载流子数密度（m⁻³）
* (e)：元电荷（1.6×10⁻¹⁹ C）
* (\bar v)：平均漂移速度（drift velocity）
* (A)：导体截面积

**漂移速度与电场：**

[
\bar v = \frac{eE\tau}{m}
]

* (E)：电场强度（V/m）
* (\tau)：平均碰撞时间（mean free time）
* (m)：电子质量

---

### 3.2 电导率/电阻率 & Ohm 定律

**电导率（conductivity）(\sigma)**

[
J = \sigma E
]

Drude 模型推导： 

[
\sigma = \frac{n e^2 \tau}{m}
]

**电阻率（resistivity）(\rho)**

* (\rho = 1/\sigma)
* 器件电阻：
  [
  R = \rho \frac{\ell}{A}
  ]

  * (\ell)：导体长度

**Ohm’s Law 欧姆定律**

[
V = IR,\quad J = \sigma E
]

---

### 3.3 载流子类型 & 能带联系

* Metals：载流子主要是 conduction electrons 自由电子（(E_F) 穿过能带）
* Semiconductors：电子 + 空穴（holes）
* Insulators：带隙大，自由载流子非常少

**常见考法**

* 解释为什么金属导电好、聚合物/陶瓷多为绝缘体（结合能带和载流子密度）
* 根据各类材料电导率数量级排序（金属 > 半导体 > 绝缘体）

---

### 3.4 Drude 模型的局限 & 超导/隧穿

* Drude 把电子当做经典气体：忽略电子-电子与电子-离子间的强相互作用 
* 无法解释：

  * 超导（superconductivity）：0 电阻，Cooper pairs
  * 量子隧穿（quantum tunneling）：在势垒下穿过绝缘层 

---

## 4. 光学性质（Week6 – 重点）

### 4.1 光的波粒二象性 & 基本关系

**基本公式**

* 频率–波长–光速：
  [
  c = \lambda \nu
  ]
* 光子能量：
  [
  E = h\nu = \frac{hc}{\lambda}
  ]

  * (h)：普朗克常数
  * (c)：真空光速（3×10⁸ m/s）
  * (\lambda)：波长
  * (\nu)：频率

**可能考：**给定 Eg，求最短吸收波长 (\lambda_{\min} = hc/E_g)（课件例：Ge、Si 计算）。

---

### 4.2 折射率与折射/反射

**折射率 Refractive index**

[
n = \frac{c}{v}
]

* (v)：光在介质中的相速度

**Snell 定律（Snell’s law）**

[
n_1 \sin\theta_1 = n_2 \sin\theta_2
]

* (\theta_1)：入射角
* (\theta_2)：折射角

**全反射条件**

* 当从高折射率介质射向低折射率介质（(n_1>n_2)）：
  [
  \sin\theta_c = \frac{n_2}{n_1}
  ]

---

### 4.3 吸收：Beer–Lambert Law + 带隙

**Beer-Lambert 定律**（Beer-Lambert law）

[
I(x) = I_0 e^{-\beta \ell}
]

* (I_0)：入射光强
* (I(x))：透射到厚度 (\ell) 后光强
* (\beta)（或记作 (\alpha)）：吸收系数（absorption coefficient, m⁻¹）

**符号说明**

* (\beta)：吸收系数（越大表示材料对该波长吸收越强）
* (\ell)：透过路径长度/厚度

**带隙与吸收条件**（重复强调，因为很爱考）

[
h\nu > E_g \quad \Rightarrow\quad \text{可以发生价带→导带跃迁，吸收光}
]

* (E_g < 1.8\ \text{eV})：可见光几乎全被吸收 → 不透明
* (E_g > 3.1\ \text{eV})：对可见光几乎全透明 → 无色透明
* 中间：部分波长吸收 → 呈现颜色（ruby/sapphire 案例）

---

### 4.4 金属 vs 介电材料的光学行为

**金属（Metals）**

* 有“电子海” sea of electrons，自由电子在可见光频率下可被驱动振荡
* 入射光 → 驱动自由电子振荡 → 重新辐射 EM 波 → 强反射
* 高频（如可见）区域仍有一定吸收 → 金属呈现有色/光泽

**介电材料（Dielectrics）**

* 没有自由电子，但有束缚电子/偶极子
* 光的电场驱动极化 polarization：偶极子微小位移
* 光频率与偶极响应频率关系 → 决定折射率、吸收峰（颜色）

**介电常数与频率**：

* Permittivity (\varepsilon)、相对介电常数 (\varepsilon_r) 与频率相关
* 不同频段（静电、低频电场、可见光）材料表现不同 

---

### 4.5 激光 LASER（可能是简答题）

**LASER = Light Amplification by Stimulated Emission of Radiation**

关键概念：

* Stimulated emission 受激辐射
* Spontaneous emission 自发辐射
* Population inversion 反转粒子数：激发态粒子数 > 基态粒子数 
* Gain medium 增益介质：Ti:sapphire, He-Ne, CO₂ 等

**可能考点：**

* 区分自发辐射 vs 受激辐射
* 为什么需要粒子数反转才能实现激光

---

## 5. 力学性质 1：金属/一般固体（Week8 – 应力应变）

### 5.1 基本应力/应变定义

**工程应力（engineering stress）**

[
\sigma = \frac{F}{A_0}
]

* (\sigma)：正应力（Pa 或 MPa）
* (F)：轴向载荷（N）
* (A_0)：初始截面积

**工程应变（engineering strain）**

[
\varepsilon = \frac{\Delta L}{L_0}
]

* (\varepsilon)：应变（无量纲）
* (L_0)：初始长度
* (\Delta L = L - L_0)：伸长量

**剪应力 & 剪应变**

[
\tau = \frac{F}{A_0},\quad \gamma = \frac{\Delta x}{h} = \tan\theta
]

* (\tau)：剪应力
* (\gamma)：剪应变

---

### 5.2 Hooke 定律 &弹性模量

**线弹性区：**

[
\sigma = E\varepsilon
]

* (E)：Young’s modulus 杨氏模量/弹性模量（Pa）

  * 反映材料刚度（stiffness），E 越大材料越“硬”（难拉伸）

**剪切弹性：**

[
\tau = G\gamma
]

* (G)：剪切模量 shear modulus

**Poisson 比（泊松比）**

[
\nu = -\frac{\varepsilon_\text{lateral}}{\varepsilon_\text{longitudinal}}
]

* 描述在拉伸方向伸长时，横向收缩的比例

**各模量关系（各向同性材料）：**

[
E = 2G(1+\nu)
]

---

### 5.3 真应力/真应变（True stress/strain）

在大变形（颈缩前）更准确：

[
\sigma_T = \frac{F}{A_\text{instant}},\quad \varepsilon_T = \ln(1+\varepsilon)
]

工程应力-应变常用于设计；真应力-应变用于材料本构分析。

---

### 5.4 屈服、强度与韧性等力学参数

**屈服强度（yield strength）(\sigma_y)**

* 金属中常用 0.2% offset method（0.2% 偏移法）确定屈服点

**抗拉强度（ultimate tensile strength, UTS）**

* 拉伸曲线中的最大应力，用于表示强度

**断后延伸率（ductility）**

[
%EL = \frac{L_f - L_0}{L_0} \times 100%
]

**截面收缩率**

[
%RA = \frac{A_0 - A_f}{A_0} \times 100%
]

* 反映塑性/延展性（ductility）

**弹性恢复能（韧度之一：模量韧性 Modulus of resilience）**

[
U_r \approx \frac{\sigma_y^2}{2E}
]

* 单位体积可弹性储存的能量（弹性区应力-应变曲线下的面积）

**断裂韧性（toughness）**

* 整个应力-应变曲线下的面积（直到断裂）
* 反映材料吸收能量而不发生断裂的能力

---

### 5.5 影响因素（温度、应变速率、缺陷等）

* 温度升高：通常 E↓，(\sigma_y)、UTS↓，延性↑
* 拉伸速率升高：常使材料“更脆”
* 缺陷（位错、空位等）：加强/削弱强度（冷加工增加位错 → 强化）

---

## 6. 力学性质 2：聚合物与相变（Week8/9 – 重点）

### 6.1 聚合物基本概念

**术语**

* Polymer 聚合物
* Monomer 单体
* Repeat unit 重复单元
* Molecular weight 分子量 (M)
* Degree of polymerization 聚合度 DP = 分子量/单体分子量

**分类**

* Thermoplastic 热塑性
* Thermoset 热固性
* Elastomer 弹性体
* Amorphous polymer 非晶聚合物
* Semi-crystalline polymer 半结晶聚合物

---

### 6.2 玻璃态、橡胶态与半结晶（Tg 与 Tm）

**玻璃化转变温度 Tg（glass transition temperature）**

* 液态聚合物**快速冷却**，来不及结晶，分子运动被冻结 → 玻璃态（glassy state）
* Tg 以下：链段运动被“冻住”，材料硬脆（如 PS at room T）

**熔点 Tm（melting temperature）**

* 缓慢冷却时，聚合物可结晶，特定温度发生体积突变 → 熔点 Tm（半结晶聚合物）

**三种冷却结果（Week9 图）：**

1. 形成 **完全非晶玻璃态**（rapid quench）
2. 形成 **半结晶 solid**（部分晶区 + 部分非晶）
3. 理想**完全结晶**（实验室极端条件才可能）

**半结晶聚合物同时具有 Tm 与 Tg：**

* Tg 以下：玻璃态
* Tg–Tm：非晶部分变橡胶态，晶区仍为固体
* Tm 以上：整体熔融

**Tg 与 Tm 的用途**

* Amorphous thermoplastic：上限使用温度靠 Tg
* 高结晶度 polymer：上限使用温度靠 Tm
* Semi-crystalline：两者都重要（橡胶态+晶体共存）

---

### 6.3 Tg 与 Tm 的影响因素（要背！）

课件总结：

* **链刚度越大 → Tg 和 Tm 都上升**
* 增加链刚度的结构特征：

  1. Bulky side groups 大体积侧基
  2. Polar groups/side groups 极性基团
  3. Chain double bonds & aromatic groups 链上的双键、芳环
* 经验：
  [
  0.5 T_m < T_g < 0.8 T_m \ (\text{Kelvin 标度})
  ]

**考试典型问法**

* 给出几个结构式，让你比较 Tg 和 Tm 大小（谁有芳香环、谁更刚）
* 问“为什么玻璃态比橡胶态硬？”（链段运动受限，自由体积小）

---

### 6.4 半结晶聚合物的双相结构

* bulk crystallinity 体积分数：如 PET ~30%，PE 可到 ~90%
* 半结晶聚合物可以看成**晶相 + 非晶相两相连通**系统，多数链同时穿过两个区域 
* 机械/扩散等性质强烈依赖于晶区与非晶区比例、晶粒尺寸等

---

### 6.5 相变的其它触发因素（Week9 尾部）

* pH & ionic strength：影响多电解质聚合物的构象/溶解度 
* Light：光敏聚合物，特定波长可触发相转变 
* Electric/magnetic field：可使链取向或相变
* Mechanical stress：机械载荷可诱导结晶（应力诱导结晶）

**典型例子：LCST（下临界溶解温度）**

* 如 PNIPA（Poly(N-isopropylacrylamide)）在水溶液中 T<LCST 时亲水可溶，T>LCST 时疏水析出 
* 自由能：(\Delta G = \Delta H - T\Delta S)：低温焓项主导，高温熵项主导 → 相行为改变

---

## 7. 扩散（Week10 – 重点）

### 7.1 扩散通量与 Fick 第一定律（Steady-state）

**扩散通量（diffusion flux）**

[
J = \frac{1}{A} \frac{\mathrm{d}M}{\mathrm{d}t}
]

* (J)：扩散通量（M/(m²·s)）
* (M)：通过面积 A 的扩散质量
* (A)：截面积
* (t)：时间

**Fick 第一定律（steady state）** 

[
J = -D \frac{dC}{dx}
]

* (D)：扩散系数（diffusion coefficient, m²/s）
* (C(x))：浓度（M/m³）
* (x)：位置坐标
* 负号：扩散从**高浓度 → 低浓度**

**与欧姆定律类比：**

* (J) ↔ 电流密度 (J_e)
* (-\dfrac{dC}{dx}) ↔ 电场 (E = -dV/dx)
* (D) ↔ 电导率 (\sigma) 

---

### 7.2 Fick 第二定律（非稳态）

[
\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}
]

* 描述随时间演化的扩散过程（浓度随 t 和 x 的变化）

**典型边界条件（半无限固体，表面浓度维持在 (C_s)）：** 

[
\frac{C(x,t) - C_0}{C_s - C_0} = 1 - \operatorname{erf}\left(\frac{x}{2\sqrt{Dt}}\right)
]

* (C_0)：初始均匀浓度
* (C_s)：表面维持浓度
* (\operatorname{erf})：误差函数

**理解要点**

* 扩散“前沿”随 (\sqrt{Dt}) 推进：(x \propto \sqrt{Dt}) 
* 许多实际扩散（渗碳、电池离子扩散等）浓度分布都符合该形式 

---

### 7.3 扩散系数的温度依赖（Arrhenius 公式）

**Arrhenius 形式** 

[
D = D_0 \exp\left(-\frac{Q_d}{RT}\right)
]

* (D_0)：前因子（m²/s）
* (Q_d)：扩散激活能（J/mol 或 kJ/mol）
* (R)：气体常数（8.314 J/(mol·K)）
* (T)：温度（K）

**对数形式：**

[
\ln D = \ln D_0 - \frac{Q_d}{R}\frac{1}{T}
]

* 常考：给若干 (T, D) 点，画 lnD vs 1/T，求斜率 → Qd

---

### 7.4 扩散应用：半导体掺杂（doping）

* 通过高温扩散将 As（5 价）或 B（3 价）扩入 Si 中

  * As → n-type（提供电子）
  * B → p-type（提供空穴）
* 典型步骤：表面沉积富 P 层 → 高温退火 → P 扩散进入 Si 形成掺杂区

---

### 7.5 Brownian motion & 扩散标度关系

Einstein 对 Brownian motion 的结果：

[
\mathbb{E}[x^2] = 2 D t
]

* RMS 位移 (\sqrt{\mathbb{E}[x^2]} \propto \sqrt{t})
* 对比 ballistic（抛体/光）：位移 (x \propto t)

**考试常见问法**

* “扩散型传输 vs 直线型（ballistic）传输”区别：

  * diffusive：随机碰撞，多次转向，x ~ √t
  * ballistic：几乎无散射，x ~ vt

---

## 8. 小结：几组重要“类比”和思维图

1. **Drude vs Fick vs Fourier（电导–扩散–热导）**

* 电导：(J_e = \sigma E)
* 质量扩散：(J_m = -D \dfrac{dC}{dx})
* 热扩散（Fourier 定律）：(q = -k \dfrac{dT}{dx})

> 共通点：
>
> * 通量 ∝ 驱动力梯度
> * 材料常数：(\sigma, D, k) 决定“传输快慢”

2. **能带 Eg → 电学 + 光学**

* Eg 小 → 金属或窄带半导体 → 高导电 → 可见光多被反射/吸收（不透明）
* 中等 Eg → 部分可见光被吸收 → 有色
* Eg 大 → 绝缘体/宽带隙 → 对可见光透明无色

3. **聚合物结构 → Tg/Tm → 力学行为**

* 刚链 + 强相互作用（芳环/极性侧基）→ Tg/Tm 高 → 高温仍较硬
* 线性柔性链 + 少侧基 → Tg/Tm 低 → 室温下可能是橡胶态

---

如果你愿意，下一步我可以：

* 把这份内容再**整理成一个可下载的 .md 或 .pdf 版“公式&名词表”**，按「公式/符号/中英文释义/典型题型」做表格化，这样考试时查起来会更快。
