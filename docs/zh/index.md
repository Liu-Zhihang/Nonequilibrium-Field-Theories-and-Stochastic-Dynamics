这是课程 [**非平衡场论与随机动力学** **Nonequilibrium Field Theories and Stochastic Dynamics, Prof. Erwin Frey, LMU Munich, Summer Semester 2025**](https://www.theorie.physik.uni-muenchen.de/lsfrey/teaching/index.html)的自学笔记。[**Erwin Frey 教授**](https://www.theorie.physik.uni-muenchen.de/lsfrey/members/group_leaders/erwin_frey/index.html)喜欢在黑板上写板书，为记录学习过程，将所学笔记整理为文，并撰写python代码加深理解。



![课程链接：https://www.youtube.com/watch?v=-pEPKnuN1iY&list=PL2IEUF-u3gRdSbgtuqH5RNTuT798s0GqX](../assets/images/remote/977886f7-e278-42d9-91f9-f61731ec8b97-0c84ad685f.png)

**注：**需要说明的是，本人只在Youtube上学习该课程，并没有讲义，都是先听课做笔记，再撰文章，原始笔记示例如下：

![个人笔记示例](../assets/images/remote/18cc2947-7f4d-40d0-ab59-51347e927534-cc2cae9ca4.png)

**课程播放列表：** [YouTube 播放列表](https://www.youtube.com/watch?v=-pEPKnuN1iY&list=PL2IEUF-u3gRdSbgtuqH5RNTuT798s0GqX)

[哔哩哔哩链接](https://www.bilibili.com/video/BV18X8wzSEa3/?spm_id_from=333.1387.0.0)

**官方课程链接：** [LMU Munich - Nonequilibrium Field Theories and Stochastic Dynamics](https://lsf.verwaltung.uni-muenchen.de/qisserver/rds?state=verpublish&status=init&vmfile=no&publishid=1075902&moduleCall=webInfo&publishConfFile=webInfo&publishSubDir=veranstaltung)


## 课程大纲

![课程大纲](../assets/images/Course_Outline.jpg)


## 课程简介


这门课介绍非平衡场论和随机动力学的基本原理和高级概念，重点关注粒子和场系统中的随机过程，特别是朗之万方程、福克-普朗克方程和路径积分等数学形式。

内容涵盖从基础随机过程理论到现代非平衡场论方法的完整知识体系，包括动态泛函和涨落定理等高级理论方法。特别关注Kardar-Parisi-Zhang方程，该方程描述了非平衡生长过程的普适性特征。此外，还将探讨活性物质系统，其中集体行为从非平衡相互作用中涌现。

应用领域包括软物质物理、活性物质和非平衡统计力学，为理解从生物过程到湍流流体等复杂系统提供理论基础。学习路径从基础随机模型开始，逐步深入马尔可夫过程的基本方程，最终探索包括动态泛函和涨落定理在内的高级场论方法。

笔记分为四个部分，共计四十讲：

**第一部分：随机过程基础 (Foundations of Stochastic Processes)** 建立了描述随机性的数学语言。从**随机游走与布朗运动 (Random Walks and Brownian Motion)** 开始，我们学习伯努利/高斯游走和扩散方程等基本模型。**基础随机模型 (Elementary Stochastic Models)** 引入泊松过程、生灭动力学和分子马达等具体例子。**马尔可夫过程与主方程 (Markov Processes and Master Equations)** 则建立了Chapman-Kolmogorov方程和离散/连续状态空间的统一框架。

- [1. 课程导论](notes/1.%20课程导论.md)
- [2. 简单随机游走](notes/2.%20简单随机游走.md)
- [3. 高斯随机游走与泊松过程](notes/3.%20高斯随机游走与泊松过程.md)
- [4. Gillespie 算法、主方程、生成函数与种群动力学](notes/4.%20Gillespie%20算法、主方程、生成函数与种群动力学.md)
- [5. 种群动态学：线性死亡过程与Lotka-Volterra 系统](notes/5.%20种群动态学：线性死亡过程与Lotka-Volterra%20系统.md)
- [6. 马尔可夫过程的基本方程：查普曼-科尔莫戈罗夫方程](notes/6.%20马尔可夫过程的基本方程：查普曼-科尔莫戈罗夫方程.md)
- [7. 前向主方程与Q矩阵](notes/7.%20前向主方程与Q矩阵.md)
- [8. 佩龙-弗罗贝尼乌斯定理、稳态与细致平衡](notes/8.%20佩龙-弗罗贝尼乌斯定理、稳态与细致平衡.md)
- [9. 非平衡态：不可逆性与熵产生的推论](notes/9.%20非平衡态：不可逆性与熵产生的推论.md)
- [10. 埃伦费斯特模型、熵与KL散度](notes/10.%20埃伦费斯特模型、熵与KL散度.md)

**第二部分：粒子的随机动力学 (Stochastic Dynamics of Particles)** 将抽象的数学工具应用于具体的物理对象。**朗之万与福克-普朗克方程 (Langevin and Fokker-Planck Equations)** 处理路径积分和乘性噪声问题。**随机模拟技术 (Stochastic Simulation Techniques)** 介绍Gillespie算法和随机积分方法。**随机热力学 (Stochastic Thermodynamics)** 则在轨迹层面讨论熵产生、细节平衡和涨落定理。

- [11. 连续马尔可夫过程与福克-普朗克方程](notes/11.%20连续马尔可夫过程与福克-普朗克方程.md)
- [12. 布朗运动与奥恩斯坦-乌伦贝克过程](notes/12.%20布朗运动与奥恩斯坦-乌伦贝克过程.md)
- [13. 作为随机过程的蒙特卡洛采样](notes/13.%20作为随机过程的蒙特卡洛采样.md)
- [14. 哈密尔顿蒙特卡洛采样](notes/14.%20哈密尔顿蒙特卡洛采样.md)
- [15. 趋化性、跑动-翻滚运动与Keller-Segel模型](notes/15.%20趋化性、跑动-翻滚运动与Keller-Segel模型.md)
- [16. Schnitzer模型、反常扩散与运动诱导相分离](notes/16.%20Schnitzer模型、反常扩散与运动诱导相分离.md)
- [17. 朗之万方程、布朗粒子与涨落-耗散定理](notes/17.%20朗之万方程、布朗粒子与涨落-耗散定理.md)
- [18. 福克-普朗克方程与斯摩棱霍夫斯基方程：从随机轨迹到概率动力学](notes/18.%20福克-普朗克方程与斯摩棱霍夫斯基方程：从随机轨迹到概率动力学.md)
- [19. 随机过程的路径积分表述](notes/19.%20随机过程的路径积分表述.md)
- [20. 随机微分方程](notes/20.%20随机微分方程.md)
- [21. 伊藤积分与统一的随机过程框架](notes/21.%20伊藤积分与统一的随机过程框架.md)
- [22. 含乘性噪声系统的路径积分](notes/22.%20含乘性噪声系统的路径积分.md)

**第三部分：从离散态到场 (From Discrete States to Fields)** 实现了关键的抽象跃迁。**反应网络与场论 (Reaction Networks and Field Theories)** 通过主方程和Kramers-Moyal展开将离散过程连续化。**场的粗粒化动力学 (Coarse-Grained Dynamics of Fields)** 研究弛豫动力学以及守恒场与非守恒场的区别。

- [23. 从粗粒化到连续场论涨落动力学](notes/23.%20从粗粒化到连续场论涨落动力学.md)
- [24. 昂萨格系数、倒易关系与动态涨落-耗散定理](notes/24.%20昂萨格系数、倒易关系与动态涨落-耗散定理.md)
- [25. 梯度动力学、相变与弛豫](notes/25.%20梯度动力学、相变与弛豫.md)
- [26. 临界慢化、动态响应与守恒律](notes/26.%20临界慢化、动态响应与守恒律.md)
- [27. 简单流体、无摩擦流体与欧拉方程的流体动力学](notes/27.%20简单流体、无摩擦流体与欧拉方程的流体动力学.md)
- [28. 粘性流体、纳维-斯托克斯方程、熵平衡与热传导](notes/28.%20粘性流体、纳维-斯托克斯方程、熵平衡与热传导.md)
- [29. 不可逆线性热力学与干性扩散粒子系统](notes/29.%20不可逆线性热力学与干性扩散粒子系统.md)
- [30. 悬浮在流体中的布朗粒子 — H模型](notes/30.%20悬浮在流体中的布朗粒子%20—%20H模型.md)

**第四部分：非平衡系统的场论 (Field Theories of Nonequilibrium Systems)** 运用最先进的理论工具。**动态泛函与MSR形式 (Dynamical Functionals and MSR Formalism)** 发展Onsager-Machlup和Janssen-de Dominicis路径积分方法。**流体动力学与涨落流体 (Hydrodynamics and Fluctuating Fluids)** 应用模型H、动量守恒和悬浮动力学。**非平衡模式形成 (Nonequilibrium Pattern Formation)** 研究失稳和非平衡稳态。**活性物质场论 (Field Theory of Active Matter)** 则探讨自推进和自组织现象。

- [31. 动态泛函、加性噪声场论与Onsager-Machlup泛函](notes/31.%20动态泛函、加性噪声场论与Onsager-Machlup泛函.md)
- [32. Janssen-De Dominicis 响应泛函与涨落-耗散关系](notes/32.%20Janssen-De%20Dominicis%20响应泛函与涨落-耗散关系.md)
- [33. 非平衡功与涨落定理](notes/33.%20非平衡功与涨落定理.md)
- [34. 有向渗流、吸收态与谱方法](notes/34.%20有向渗流、吸收态与谱方法.md)
- [35. 主方程的路径积分表示](notes/35.%20主方程的路径积分表示.md)
- [36. 相干态路径积分、算符代数与虚噪声](notes/36.%20相干态路径积分、算符代数与虚噪声.md)
- [37. Kramers-Moyal 展开与路径积分的低噪声极限](notes/37.%20Kramers-Moyal%20展开与路径积分的低噪声极限.md)
- [38. 多物种路径积分与循环竞争动力学](notes/38.%20多物种路径积分与循环竞争动力学.md)
- [39. 从粒子跳跃到连续场论](notes/39.%20从粒子跳跃到连续场论.md)
- [40. 统一的场论框架](notes/40.%20统一的场论框架.md)

## 使用说明

每个 Python 文件对应于讲座系列中涵盖的特定主题。这些代码是根据 YouTube 视频中介绍的理论概念而实现的，作为自学和学习笔记的一部分而开发。

以下是一些代码输出演示：

![code/5.PhaseDiagram.py](../assets/images/remote/a7249f6d-3693-4c80-b256-49f91f882052-06432353d7.png)

![code/13.MCMC_LotkaVolterra.py](../assets/images/remote/75e9b1b2-4ff7-4b3f-aa6c-7a7c1988bb4d-43bd03f19f.png)

![code/16.MIPS.py](../assets/images/remote/48116ba2-bc10-4987-b72a-17b84e6e4fcb-3eade8b3cf.gif)


![code/19.OverdampedLangevinEquation.py](../assets/images/remote/c65ac76c-2634-45ca-8c51-b0578a929cf3-982dfc9b72.png)


![code/26.CriticalSlowingDown1.py](../assets/images/remote/critical_slowing_down.gif)


<video src="../assets/images/remote/fluid_simulation.mp4" controls="controls" style="max-width: 100%;"></video>
<p><strong>code/27.Eulerian_Fluid_Simulation.py</strong></p>



![code/28.VortexDecayDemo.py](../assets/images/remote/vortex_decay_detailed.gif)


![code/30.AdvectionAndDiffusion.py](../assets/images/remote/brownian_particles_comet.gif)

![code/35.Schlöglmodel.py](../assets/images/remote/schlogl_bistability_evolution.gif)

![code/37.PercolationFieldTheory.py](../assets/images/remote/directed_percolation_field.gif)

![code/38.Rock-Paper-Scissors.py](../assets/images/remote/spatial_rps_spirals.gif)


![code/39.ReactionFront.py](../assets/images/remote/reaction_front_comparison.gif)

<video src="../assets/images/remote/Starling.mp4" controls="controls" style="max-width: 100%;"></video>
<p><strong>Starling Simulation</strong></p>



## 学习所需基础

- 统计力学与热力学
- 概率论与随机过程
- 微分方程
- 场论基础知识（有帮助但非必需）

## 许可证

本项目采用 [CC BY-NC-ND 4.0](../../LICENSE) 许可证。

## 致谢

- 感谢慕尼黑大学 Erwin Frey 教授及其 Physics of Life 研究组提供的精彩课程
- 感谢 [PhysicsOfLifeLMU YouTube 频道](https://www.youtube.com/@PhysicsOfLifeLMU) 公开发布这些课程

## Citation（引用）

若本仓库对您有帮助，可使用如下 BibTeX 格式引用：

```bibtex
@misc{liu2025nonequilibrium,
  author = {Liu, Zhihang},
  title = {Nonequilibrium Field Theories and Stochastic Dynamics: Course Notes and Code},
  year = {2025},
  url = {https://github.com/Liu-Zhihang/Nonequilibrium-Field-Theories-and-Stochastic-Dynamics},
  note = {Self-study notes for the course by Prof. Erwin Frey, LMU Munich}
}
```