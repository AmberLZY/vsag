# HGraph BFS 图重排优化方案

## 1. 背景

### 1.1 Topdown 数据

```
Backend Bound = 80.53%
  Memory Bound = 65.91%  ← 主瓶颈：数据访问延迟
  Core Bound   = 14.62%
```

65.91% Memory Bound 意味着 CPU 大量时间在等数据。HGraph 搜索 = 图邻居跳转（间接访存）+ 向量距离计算（顺序访存）。图遍历中跨节点跳转的 cache miss 是核心瓶颈。

### 1.2 BFS 图重排原理

HGraph 的图构建按节点插入顺序（通常是随机顺序）排列 inner_id。搜索时从入口点沿边跳转，相邻访问的节点在内存中可能相距很远，无法受益于 cache 预取和空间局部性。

BFS 重排：从入口点做 BFS 遍历，将节点按 BFS 访问顺序重新排列——搜索时大概率按同一条 BFS 顺序访问，相邻节点在内存上也相邻，L2/L3 cache hit 率提升。

### 1.3 参考实现

FAISS KRL `IndexHNSW::graphBFSPerm()`（`IndexHNSW.cpp:82-122`），改造 HNSW 图节点顺序:
- 效果：ARM 上 HNSW 搜索 QPS 提升 10-15%
- 前提：FAISS 使用连续数组存储邻接表，重排成本低

## 2. 影响分析

### 2.1 数据结构

```
内/外部ID映射
```

VSAG HGraph 内部使用 `inner_id`（密集 0..N-1）索引所有组件：

```
inner_id
├── label_table_[inner_id] → label (外部 ID)
├── basic_flatten_codes_[inner_id] → 量化向量
├── high_precise_codes_[inner_id] → 高精度向量
├── raw_vector_[inner_id] → 原始 FP32 向量
├── bottom_graph_.neighbors_[inner_id] → 邻接列表 (Vector<InnerIdType>)
├── route_graphs_[i].neighbors_[inner_id] → 路由图邻接
├── extra_infos_[inner_id] → 额外信息
├── entry_point_id_ → BFS 起点
├── neighbors_mutex_[inner_id] → 锁
└── pool_.visited_list[inner_id] → 访问标记
```

**所有组件通过 inner_id 索引，重排就是重新映射 inner_id。**

### 2.2 与 FAISS 的关键差异

| | FAISS HNSW | VSAG HGraph |
|---|---|---|
| 邻接表存储 | **连续数组** (`offsets[]` + `neighbors[]`) | **哈希映射** (`UnorderedMap<id, Vector<id>>`) |
| 向量存储 | 连续数组 | 连续（可批量搬移） |
| 重排成本 | 低（swap 数组元素） | **高**（哈希表 erase + insert） |
| 批量接口 | 无，但数组操作天然高效 | 无，哈希映射操作昂贵 |

这是工程难点：VSAG 的 `SparseGraphDataCell` 使用哈希映射，需要逐个 erase/insert，不能像 FAISS 那样直接 swap。

### 2.3 为什么在 Build 时做而非 Load 时

FAISS KRL 的 `graphBFSPerm()` 是在 `index_read.cpp`（反序列化）中调用的，属于 **load 模式**。VSAG 选择 **build 模式**，原因如下：

**FAISS 为什么在 load 时做：**

KRL 是 FAISS 的外部补丁。FAISS 的序列化格式是标准化的（`fourcc("IHNp")` 等），KRL 不能修改——改了就不兼容标准 FAISS。所以唯一能插入重排的位置就是加载完之后、搜索之前。好在 FAISS 的 `permute_entries` 只是 swap 连续数组元素，O(N)，代价很低。

```
FAISS load 模式:
read_index(f) → 读出标准 FAISS 格式 → graphBFSPerm() → permute_entries() → 搜索
                 ↑ 格式不能动，只能后处理
```

**VSAG 为什么应该在 build 时做：**

| 维度 | FAISS load 模式 | VSAG build 模式 |
|---|---|---|
| 存储结构 | 连续数组，swap O(1) | 哈希映射，erase+insert 昂贵 |
| 序列化兼容 | KRL 不能改标准 FAISS 格式 | VSAG 完全控制自己的格式 |
| 操作频率 | 每次加载都做 | 构建时做一次，保存后零开销 |
| 加载延迟 | 可接受（数组 swap 快） | 不可接受（哈希重排慢，每次启动都要等） |

核心结论：

1. **Build 时重排 → Serialize → Load 直接顺序读取**，load 路径零额外开销
2. **Load 时重排**意味着每次启动都要对哈希映射做全量 erase+insert，启动延迟不可接受
3. VSAG 控制自己的序列化格式，没有兼容负担，没必要把开销推迟到 load 路径

唯一需要 load 模式重排的场景是：希望同一个已序列化索引在不同部署下灵活开关重排。但 `HGRAPH_ENABLE_BFS_REORDER` 参数已在 build 时控制此行为，直接构建两个索引即可，不需要增加序列化格式复杂度。

### 2.4 收益评估

| 场景 | 预期 QPS 提升 | 依据 |
|---|---|---|
| ARM NEON (鲲鹏 920) | **10-15%** | 对标 FAISS BFS perm 在 ARM 上的实测收益 |
| x86 (AVX2/AVX512) | 5-10% | 常见的图遍历局部性改善，但 x86 cache 更大，收益相对较小 |
| 小数据集 (<10 万) | 0-5% | 索引全在 L3 cache，重排收益有限 |
| 大数据集 (>100 万) | **15-20%** | Memory Bound 更严重，局部性改善更显著 |

**你的场景 (Memory Bound 65.91%)：预取收益高，但 BFS 重排的作用可能被 prefetch 部分覆盖。两者是互补关系——prefetch 解决延迟，BFS 重排减少整体 miss 量。**

## 3. 劣化风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **构建时间增加** | 100% | 构建延长 10-30% | BFS 遍历 O(N+M)，重排 O(N*avg_degree) 图操作。仅构建时一次性开销，不影响搜索 |
| **增量插入/删除后局部性退化** | 中 | 长期运行后 BFS 序逐渐退化 | 定期后台重建；长周期来说 HGraph 本身就需要重建 |
| **某些搜索模式不利** | 低 | 修改 ef_search 或 filter 率后，搜索路径偏离 BFS 序 | BFS 序下相邻节点距离也接近，部分抵消；极端 filter 回退 bruteforce |
| **x86 收益不明显但 ARM 好** | 中 | 跨平台收益不均 | ARM Memory Bound 通常更高（cache 更小），这符合规律 |
| **构建耗内存翻倍** | 100% | 方案 A 需要旧+新实例共存 | 只在构建时；可选在序列化后释放旧实例 |

**结论：没有搜索 QPS 劣化的风险，主要代价是一次性构建开销。**

## 4. 实施位置

### 4.1 插入点

```
HGraph::Build(dataset)
    ├── Train(data)           ← 量化器训练
    ├── build_by_odescent()   ← 图构建
    ├── [BFS Reorder 插入点]  ← ★ 在此处
    ├── verify()              ← 已有校验
    └── Serialize()           ← 序列化（自动使用新 ID 映射）
```

**插入位置**：`src/algorithm/hgraph/hgraph_build.cpp`，`Build()` 函数中 `build_by_odescent()` 调用之后、可选校验之前。

BFS 重排完成后，`label_table_`、`basic_flatten_codes_`、`bottom_graph_` 等都已在新的 inner_id 空间下，后续 `Serialize()` 自然写出优化后的布局。

### 4.2 改动文件

| 文件 | 改动内容 |
|---|---|
| `src/algorithm/hgraph/hgraph_build.cpp` | 在 Build() 末尾调用 BFS 重排函数 |
| `src/algorithm/hgraph/hgraph.cpp` | 新增 `ReorderByBFS()` 函数 |
| `src/algorithm/hgraph/hgraph.h` | 声明 `ReorderByBFS()` |
| `src/algorithm/hgraph/hgraph_parameter.h` | 新增 `enable_bfs_reorder` 参数 |
| `src/algorithm/hgraph/hgraph_parameter.cpp` | 参数校验 |
| `src/algorithm/hgraph/hgraph_param_mapping.cpp` | JSON 参数映射 |

### 4.3 不改动

- `FlattenInterface` / `GraphInterface` — 不需新增接口（方案 A）
- `serialize.cpp` — 自动使用新 ID 映射
- `hgraph_search.cpp` — 搜索逻辑完全不变
- 所有数据格式 — 序列化格式不变（只是节点顺序不同）

## 5. 算法设计

### 5.1 BFS 遍历生成 Permutation

```cpp
// BFS 遍历策略（参考 FAISS graphBFSPerm）
ReorderByBFS() {
    InnerIdType n = total_count_;
    Vector<InnerIdType> perm;     // perm[new_id] = old_id
    Vector<InnerIdType> imap;     // imap[old_id] = new_id
    Vector<bool> visited(n, false);
    std::queue<InnerIdType> q;

    // Step 1: 从高 level 节点开始 BFS
    for (auto& rg : route_graphs_) {
        for (auto& [id, neighbors] : rg->GetAllNeighbors()) {
            if (!visited[id]) { q.push(id); visited[id] = true; }
        }
    }
    // 如果路由图为空，从 entry_point 开始
    if (q.empty()) { q.push(entry_point_id_); visited[entry_point_id_] = true; }

    // Step 2: BFS 遍历
    while (!q.empty()) {
        InnerIdType v = q.front(); q.pop();
        perm.push_back(v);
        for (auto& neighbor : bottom_graph_->GetNeighbors(v)) {
            if (!visited[neighbor]) { q.push(neighbor); visited[neighbor] = true; }
        }
    }

    // Step 3: 剩余未访问节点附加在末尾
    for (InnerIdType i = 0; i < n; ++i) {
        if (!visited[i]) perm.push_back(i);
    }

    // Step 4: 构建反向映射 imap
    imap.resize(n);
    for (InnerIdType new_id = 0; new_id < n; ++new_id) {
        imap[perm[new_id]] = new_id;
    }

    ApplyPermutation(perm, imap);
}
```

### 5.2 重排具体数据结构

```cpp
ApplyPermutation(perm, imap) {
    // === 1. 向量编码 (最大开销) ===
    // 建立新的 FlattenInterface 实例，按 perm 顺序写入
    auto new_basic = FlattenInterface::MakeInstance(param, common_param);
    auto new_high  = FlattenInterface::MakeInstance(param, common_param);
    // ... 同样处理 raw_vector_
    for (new_id = 0..n-1) {
        old_id = perm[new_id];
        new_basic->CopyFrom(*basic_flatten_codes_, old_id);
        new_high ->CopyFrom(*high_precise_codes_,  old_id);
    }
    swap(basic_flatten_codes_, new_basic);
    swap(high_precise_codes_,  new_high);

    // === 2. 图边 (核心代码) ===
    // 建立新的 GraphInterface 实例
    // 遍历每个 old_id 的邻居列表，将邻居也通过 imap 转换
    auto new_bottom = GraphInterface::MakeInstance(param);
    for (new_id = 0..n-1) {
        old_id = perm[new_id];
        auto old_neighbors = bottom_graph_->GetNeighbors(old_id);
        Vector<InnerIdType> new_neighbors;
        for (auto nb : old_neighbors) {
            new_neighbors.push_back(imap[nb]);  // 邻居 ID 也要转换！
        }
        new_bottom->SetNeighbors(new_id, new_neighbors);  // 需要添加 SetNeighbors 方法
    }
    swap(bottom_graph_, new_bottom);
    // route_graphs_ 同样处理

    // === 3. 简单映射 ===
    // label_table_: 建立新 table，按 perm 重排
    // extra_infos_: 同上
    // attr_filter_index_: 同上

    // === 4. 单值更新 ===
    entry_point_id_ = imap[entry_point_id_];

    // === 5. 重建辅助结构 ===
    neighbors_mutex_->Resize(n);
    pool_ = VisitedListPool(n, allocator);
}
```

## 6. 需要新增的接口

当前 `GraphInterface` 和 `FlattenInterface` 没有便捷的重排方法。方案 A 需新增：

### GraphInterface 新增

```cpp
// 读取一个节点的邻居列表
virtual Vector<InnerIdType> GetNeighbors(InnerIdType id) const = 0;

// 批量设置节点的邻居（用于重建图）
// 在 SparseGraphDataCell 中实现：直接赋值到 neighbors_[id]
virtual void SetNeighbors(InnerIdType id, const Vector<InnerIdType>& neighbors) = 0;

// 遍历所有邻居（BFS 用）
virtual void ForEachNode(std::function<void(InnerIdType, const Vector<InnerIdType>&)> fn) const = 0;
```

### FlattenInterface 新增

```cpp
// 从另一个 FlattenInterface 按 old_id 复制一个向量
// FlattenDataCell 中实现：memcpy(dst + new_id * code_size, src + old_id * code_size, code_size)
virtual void CopyFrom(const FlattenInterface& src, InnerIdType src_id) = 0;
```

> **备选方案**：不新增接口，而是直接在 BFS 重排函数中使用 `SparseGraphDataCell` 的内部数据。这侵入性更大但也有先例——VSAG 内部常常用 `dynamic_cast` 访问具体实现。权衡后新增接口更干净。

## 7. 开关控制

新增可选参数，默认关闭：

```json
// hgraph_param_mapping.cpp 新增
"HGRAPH_ENABLE_BFS_REORDER": false  // 默认
```

原因：
- 构建开销：BFS 重排会增加构建时间约 10-30%
- 增量构建场景不适用（插入会破坏 BFS 序）
- 需要用户根据 Memory Bound 比例决定是否开启
- 小数据集 (<10 万) 收益可忽略不计

## 8. 与现有优化的关系

| 优化 | 与 BFS 重排的关系 |
|---|---|
| FP32 L2 NEON 4-accumulator | **互补**：BFS 重排减少 memory stall，NEON 优化利用计算窗口 |
| FP16 量化 | **互补**：FP16 减少数据量 50%，BFS 重排进一步减少 miss |
| ELP 预取优化器 | **部分重叠**：ELP 自动调 prefetch 参数，BFS 重排让 prefetch 更有效（连续页） |
| 查询预量化 | **独立**：减少每个向量的计算开销，不冲突 |

## 9. 实施步骤

| 阶段 | 内容 | 估时 |
|---|---|---|
| Phase 1 | 新增 `ReorderByBFS()` 函数和接口（`GetNeighbors`/`SetNeighbors`/`CopyFrom`） | 2-3 天 |
| Phase 2 | 在 `Build()` 中集成，添加参数开关 | 1 天 |
| Phase 3 | 单元测试 + 正确性验证（搜索召回率不变） | 2 天 |
| Phase 4 | 鲲鹏 950 性能测试（Memory Bound 变化，QPS 变化） | 1 天 |

---

## 10. 总结

| 维度 | 评估 |
|---|---|
| 可行性 | 完全可行，BFS 遍历 + 逐个搬运 |
| 工程复杂度 | **中**（主要是搬运逻辑 + 3 个新接口） |
| 搜索 QPS | **+10-20%** (ARM), **+5-10%** (x86) |
| 构建开销 | **+10-30%** 构建时间，+100% 峰值内存 |
| 劣化风险 | 搜索 QPS **无劣化**；增量插入/删除后会逐渐退化 |
| 关键前提 | Memory Bound > 50% 才值得开启；
