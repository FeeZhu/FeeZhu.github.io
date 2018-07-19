## TBB编程
TBB(Thread Building Blocks)是英特尔发布的一个库，全称为 Threading Building Blocks。
#### 为什么要TBB
在多核的平台上开发并行化的程序，必须合理地利用系统的资源 - 如与内核数目相匹配的线程，内存的合理访问次序，最大化重用缓存。有时候用户使用(系统)低级的应用接口创建、管理线程，很难保证是否程序处于最佳状态。 而 Intel Thread Building Blocks (TBB) 很好地解决了上述问题： 
  - 1）TBB提供C++模版库，用户不必关注线程，而专注任务本身。 
  - 2）抽象层仅需很少的接口代码，性能上毫不逊色。 
  - 3）灵活地适合不同的多核平台。 
  - 4）线程库的接口适合于跨平台的移植(Linux, Windows, Mac) 
  - 5）支持的C++编译器 – Microsoft, GNU and Intel  
#### TBB结构
TBB包含了 Algorithms、Containers、Memory Allocation、Synchronization、Timing、Task Scheduling这六个模块。TBB的结构：

##### 通用并行算法
① `parallel_for`
parallel_for是在一个值域执行并行迭代操作的模板函数（如对数组求和）

②`parallel_reduce`
parallel_reduce模板在一个区域迭代，将由各个任务计算得到的部分结果合并，得到最终结果。

.....

#### 流的并行算法