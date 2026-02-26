# 电影推荐系统 - 代码详解与可解释性分析

## 目录
1. [项目架构概览](#1-项目架构概览)
2. [数据预处理流程](#2-数据预处理流程)
3. [推荐算法详解](#3-推荐算法详解)
4. [可解释性分析](#4-可解释性分析)
5. [Web界面实现](#5-web界面实现)
6. [核心代码逐行解析](#6-核心代码逐行解析)

---

## 1. 项目架构概览

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户界面层                                   │
│                    (app.py - Streamlit)                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                         推荐引擎层                                   │
│                  (movie_recommender.py)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │ 基于内容    │  │ 基于元数据  │  │ 协同过滤    │  │ 混合推荐  │  │
│  │ TF-IDF      │  │ CountVec    │  │ SVD/KNN     │  │ 加权融合  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                         数据处理层                                   │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ tmdb_5000_movies.csv│  │ tmdb_5000_credits.csv               │  │
│  │ 电影元数据          │  │ 演职人员数据                        │  │
│  └─────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心文件说明

| 文件 | 行数 | 职责 |
|------|------|------|
| `movie_recommender.py` | 974行 | 推荐算法核心实现 |
| `app.py` | 495行 | Streamlit Web界面 |
| `tmdb_analysis.py` | 317行 | 数据探索与可视化 |

---

## 2. 数据预处理流程

### 2.1 数据加载与合并

```python
# movie_recommender.py: 52-59行
self.movies = pd.read_csv(movies_path)
self.credits = pd.read_csv(credits_path)

# 合并数据集 (117-119行)
self.credits.columns = ['id', 'title', 'cast', 'crew']
self.movies = self.movies.merge(self.credits[['id', 'cast', 'crew']], on='id')
```

**数据流向图：**
```
movies.csv (4803行)          credits.csv (4803行)
    │                              │
    │ id, title, genres,           │ id, title, cast, crew
    │ overview, budget...          │
    │                              │
    └──────────────┬───────────────┘
                   │ MERGE ON 'id'
                   ▼
            合并后数据集 (4803行)
            包含所有电影信息和演职人员
```

### 2.2 JSON字段解析

原始数据中 `genres`, `keywords`, `cast`, `crew` 字段是JSON格式字符串：

```python
# 原始格式示例
genres = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'

# 解析函数 (78-83行)
def _safe_literal_eval(self, x):
    """安全地解析JSON字符串"""
    try:
        return ast.literal_eval(x)  # 将字符串转为Python对象
    except (ValueError, SyntaxError):
        return []

# 解析后格式
genres = [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
```

### 2.3 特征提取

```python
# 128-137行: 提取关键特征
self.movies['genres_list'] = self.movies['genres'].apply(self._get_list)
# 输入: [{"id": 28, "name": "Action"}, ...]
# 输出: ["Action", "Adventure", ...]

self.movies['director'] = self.movies['crew'].apply(self._get_director)
# 从crew列表中找到job='Director'的人
# 输出: "James Cameron"

self.movies['cast_list'] = self.movies['cast'].apply(self._get_top_cast)
# 取前5个演员
# 输出: ["Sam Worthington", "Zoe Saldana", ...]
```

### 2.4 文本清洗

```python
# 105-111行: 标准化文本
def _clean_text(self, x):
    """清理文本，移除空格并转为小写"""
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ''

# 示例
输入: ["Science Fiction", "Action"]
输出: ["sciencefiction", "action"]

# 为什么要这样做？
# 确保 "Science Fiction" 和 "science fiction" 被视为相同特征
```

---

## 3. 推荐算法详解

### 3.1 方法一：基于内容的推荐 (Content-Based Filtering)

#### 3.1.1 算法原理

```
电影A的简介 ─────┐
                 │     TF-IDF      余弦相似度
电影B的简介 ─────┼───► 向量化 ───► 相似度矩阵 ───► 推荐列表
                 │
电影C的简介 ─────┘
```

#### 3.1.2 TF-IDF 详解

**TF (Term Frequency) - 词频：**
```
TF(词, 文档) = 该词在文档中出现次数 / 文档总词数
```

**IDF (Inverse Document Frequency) - 逆文档频率：**
```
IDF(词) = log(总文档数 / 包含该词的文档数)
```

**TF-IDF 最终值：**
```
TF-IDF = TF × IDF
```

**为什么用TF-IDF？**
- "the", "a" 等常见词的IDF很低，被自动降权
- 特征词（如 "alien", "spacecraft"）的IDF较高，被放大
- 能捕捉电影简介的**独特主题**

#### 3.1.3 代码实现解析

```python
# 153-180行
def build_content_based_model(self):
    # 创建TF-IDF向量化器
    tfidf = TfidfVectorizer(
        stop_words='english',    # 移除英语停用词 (the, is, at...)
        max_features=5000,       # 最多保留5000个特征词
        ngram_range=(1, 2)       # 使用单词和双词组合
    )

    # 向量化所有电影简介
    tfidf_matrix = tfidf.fit_transform(self.movies['overview'])
    # 结果: 稀疏矩阵 (4803电影 × 5000特征)

    # 计算相似度矩阵
    self.content_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    # linear_kernel 等价于 cosine_similarity，但更快
    # 结果: 密集矩阵 (4803 × 4803)
```

#### 3.1.4 余弦相似度计算

```
余弦相似度 = (A · B) / (||A|| × ||B||)

其中:
- A · B 是两个向量的点积
- ||A|| 是向量A的模（长度）
```

**图示：**
```
     Avatar的TF-IDF向量
         ↓
    [0.2, 0.0, 0.5, 0.1, ...]  ←── 5000维
         │
         │ 计算余弦夹角
         ▼
    [0.1, 0.3, 0.4, 0.2, ...]  ←── Interstellar的向量
         ↓
    相似度 = 0.85 (越接近1越相似)
```

### 3.2 方法二：基于元数据的推荐 (Metadata-Based)

#### 3.2.1 算法原理

```
类型 ────────┐
导演 ────────┤     特征      Count        余弦
演员 ────────┼───► 组合 ───► 向量化 ───► 相似度
关键词 ──────┘     (Soup)
```

#### 3.2.2 特征权重设计

```python
# 230-257行: 创建综合特征
def create_soup(row):
    features = []

    # 类型权重 ×3 (最重要的特征)
    features.extend(row['genres_clean'] * 3)
    # ["action", "adventure"] × 3 = ["action", "adventure", "action", "adventure", "action", "adventure"]

    # 导演权重 ×3 (导演风格很重要)
    if row['director_clean']:
        features.extend([row['director_clean']] * 3)
    # "jamescameron" × 3

    # 演员权重 ×2 (演员组合有一定影响)
    features.extend(row['cast_clean'] * 2)
    # ["samworthington", "zoesaldana", ...] × 2

    # 关键词权重 ×1 (辅助特征)
    features.extend(row['keywords_clean'])

    return ' '.join(features)
```

**为什么这样设计权重？**

| 特征 | 权重 | 理由 |
|------|------|------|
| 类型 | ×3 | 用户通常按类型选片，是最强的兴趣指标 |
| 导演 | ×3 | 导演决定电影风格，粉丝会追同导演作品 |
| 演员 | ×2 | 明星效应存在，但不如类型/导演强 |
| 关键词 | ×1 | 辅助信息，帮助区分同类型电影 |

#### 3.2.3 CountVectorizer vs TF-IDF

```python
# 为什么元数据用CountVectorizer而不是TF-IDF？
count_vectorizer = CountVectorizer(
    stop_words='english',
    max_features=10000
)
```

**原因：**
- TF-IDF 适合**自然语言文本**（如电影简介）
- 元数据特征已经是**处理好的标签**，每个出现都有意义
- 用Count直接计数，保留权重倍数的效果

### 3.3 方法三：协同过滤 (Collaborative Filtering)

#### 3.3.1 核心思想

```
"物以类聚，人以群分"

基于物品CF: 喜欢A电影的人也喜欢B电影 → A和B相似
基于用户CF: 用户X和用户Y口味相似 → 推荐Y喜欢但X没看过的
```

#### 3.3.2 用户-电影评分矩阵模拟

由于TMDB数据集没有真实用户评分，代码模拟了评分数据：

```python
# 326-414行: 模拟用户偏好
def _generate_user_movie_matrix(self, n_users=1000, sparsity=0.02):
    # 初始化 1000用户 × 4803电影 的评分矩阵
    ratings = np.zeros((n_users, n_movies))

    for user_id in range(n_users):
        # 1. 为每个用户随机分配1-3个偏好类型
        preferred_genres = np.random.choice(genre_list, size=n_preferred)

        # 2. 计算每部电影被该用户选中的概率
        movie_probs = np.ones(n_movies) * 0.1  # 基础概率

        for movie_idx in range(n_movies):
            # 如果电影类型匹配用户偏好，提高概率
            if any(g in preferred_genres for g in movie_genres):
                movie_probs[movie_idx] = 0.8  # 8倍概率

            # 高评分电影更容易被选中
            movie_probs[movie_idx] *= (vote_avg / 10.0 + 0.5)

        # 3. 生成评分 (1-5分)
        base_rating = 2.5 + (vote_avg - 5) / 2
        if 类型匹配:
            base_rating += random(0.5, 1.5)  # 喜欢的类型加分
        else:
            base_rating += random(-1.0, 0.5)  # 不喜欢的类型可能减分
```

**模拟逻辑的合理性：**
```
真实用户行为                    模拟实现
───────────────────────────────────────────────
用户有类型偏好          →      随机分配偏好类型
更可能看喜欢的类型      →      匹配类型概率×8
好电影更多人看          →      评分高的概率更大
喜欢的类型评分更高      →      匹配类型评分+0.5~1.5
```

#### 3.3.3 Item-Based 协同过滤

```python
# 449-466行
def _build_item_based_cf(self):
    # 转置矩阵：行=电影，列=用户
    movie_user_matrix = self.user_movie_matrix.T
    # (1000用户 × 4803电影) → (4803电影 × 1000用户)

    # 计算电影间相似度
    self.item_similarity_cf = cosine_similarity(movie_user_matrix)
    # 结果: (4803 × 4803) 电影相似度矩阵
```

**直观理解：**
```
            用户1  用户2  用户3  ...
Avatar      4.5    5.0    4.2
Titanic     4.8    4.5    4.0
The Matrix  3.0    2.5    4.5

如果Avatar和Titanic的评分模式相似（同一批人喜欢）
→ 它们的相似度高
→ 喜欢Avatar的人可能也喜欢Titanic
```

#### 3.3.4 SVD矩阵分解

```python
# 484-509行
def _build_svd_cf(self, n_factors=50):
    # 中心化：减去每个用户的平均评分
    ratings_centered = self.user_movie_matrix - user_ratings_mean

    # SVD分解
    U, sigma, Vt = svds(csr_matrix(ratings_centered), k=n_factors)

    # U: 用户因子矩阵 (1000用户 × 50因子)
    # sigma: 奇异值 (50,)
    # Vt: 电影因子矩阵 (50因子 × 4803电影)

    # 重构预测评分
    self.svd_predictions = U @ sigma @ Vt + user_ratings_mean
```

**SVD的直觉理解：**
```
原始评分矩阵 (1000×4803) ≈ U (1000×50) × Σ (50×50) × V^T (50×4803)

50个潜在因子可能代表：
- 因子1: 动作片偏好
- 因子2: 爱情片偏好
- 因子3: 对大导演的偏好
- ...

通过低秩近似，发现隐藏的用户-电影关系
```

### 3.4 混合推荐 (Hybrid)

```python
# 704-766行
def hybrid_recommend(self, title, top_n=10, weights=(0.3, 0.4, 0.3)):
    # 综合分数 = 加权平均
    hybrid_scores = np.zeros(n_movies)

    hybrid_scores += weights[0] * self.content_similarity[idx]   # 内容
    hybrid_scores += weights[1] * self.metadata_similarity[idx]  # 元数据
    hybrid_scores += weights[2] * self.item_similarity_cf[idx]   # 协同过滤
```

**为什么要混合？**

| 单一方法 | 优势 | 劣势 |
|----------|------|------|
| 内容 | 无冷启动，可解释 | 只能推荐相似内容 |
| 元数据 | 多维度匹配 | 需要完整元数据 |
| 协同过滤 | 发现隐藏关联 | 冷启动问题，不可解释 |

**混合推荐取长补短，提高推荐质量和鲁棒性**

---

## 4. 可解释性分析

### 4.1 什么是推荐系统的可解释性？

可解释性指：**能够向用户清晰解释"为什么推荐这部电影"**

```
不可解释: "系统推荐您观看《Inception》"
可解释:   "因为您喜欢《Interstellar》，这两部电影：
          - 都是科幻类型
          - 都由Christopher Nolan执导
          - 主题都涉及时间和意识"
```

### 4.2 各方法的可解释性对比

#### 4.2.1 基于内容 - 高可解释性 ⭐⭐⭐⭐⭐

```python
# 推荐逻辑
Avatar → 推荐 Interstellar

# 可解释性实现
1. 提取TF-IDF关键词
   Avatar: ["alien", "planet", "war", "technology", "species"]
   Interstellar: ["space", "planet", "survival", "wormhole", "time"]

2. 找到共同主题词
   共同: ["planet", "space exploration themes"]

3. 生成解释
   "推荐《Interstellar》因为它和《Avatar》都涉及外星球探索主题"
```

**增强可解释性的代码改进建议：**
```python
def explain_content_recommendation(self, source_movie, target_movie):
    """解释为什么基于内容推荐某电影"""
    # 获取两部电影的TF-IDF向量
    source_vector = self.tfidf_matrix[source_idx]
    target_vector = self.tfidf_matrix[target_idx]

    # 找到两部电影共同的高权重词
    feature_names = self.tfidf.get_feature_names_out()

    # 计算特征贡献度
    contributions = source_vector.multiply(target_vector)

    # 排序找到最重要的共同特征
    top_features = sorted(
        zip(feature_names, contributions.toarray()[0]),
        key=lambda x: x[1], reverse=True
    )[:5]

    return f"推荐原因: 共同主题词 - {[f[0] for f in top_features]}"
```

#### 4.2.2 基于元数据 - 高可解释性 ⭐⭐⭐⭐⭐

```python
# 推荐逻辑可直接解释
Avatar → 推荐 Titanic

# 元数据对比
特征          Avatar              Titanic            匹配
────────────────────────────────────────────────────────
导演          James Cameron       James Cameron      ✓ (权重×3)
类型          [Sci-Fi, Action]    [Drama, Romance]   ✗
主演          Sam Worthington     Leonardo DiCaprio  ✗
关键词        ["alien", ...]      ["ship", ...]      ✗

# 生成解释
"推荐《Titanic》因为它和《Avatar》都是 James Cameron 执导的作品"
```

**增强可解释性的代码：**
```python
def explain_metadata_recommendation(self, source_movie, target_movie):
    """解释元数据推荐原因"""
    source = self.movies[self.movies['title'] == source_movie].iloc[0]
    target = self.movies[self.movies['title'] == target_movie].iloc[0]

    reasons = []

    # 检查导演匹配
    if source['director'] == target['director']:
        reasons.append(f"同一导演: {source['director']}")

    # 检查类型重叠
    common_genres = set(source['genres_list']) & set(target['genres_list'])
    if common_genres:
        reasons.append(f"相同类型: {', '.join(common_genres)}")

    # 检查演员重叠
    common_cast = set(source['cast_list']) & set(target['cast_list'])
    if common_cast:
        reasons.append(f"共同演员: {', '.join(common_cast)}")

    # 检查关键词重叠
    common_keywords = set(source['keywords_list']) & set(target['keywords_list'])
    if common_keywords:
        reasons.append(f"相似主题: {', '.join(list(common_keywords)[:3])}")

    return reasons if reasons else ["整体特征相似"]
```

#### 4.2.3 协同过滤 - 低可解释性 ⭐⭐

```python
# 推荐逻辑难以直接解释
Avatar → 推荐 The Avengers

# 原因（用户看不懂）
"因为喜欢Avatar的用户群体中，有68%也给The Avengers打了高分"

# 更好的解释方式
"喜欢《Avatar》的观众也喜欢《The Avengers》"
```

**增强可解释性：**
```python
def explain_cf_recommendation(self, source_movie, target_movie):
    """解释协同过滤推荐"""
    source_idx = self.title_to_idx[source_movie]
    target_idx = self.title_to_idx[target_movie]

    # 找到同时喜欢两部电影的用户
    source_fans = np.where(self.user_movie_matrix[:, source_idx] >= 4)[0]
    target_fans = np.where(self.user_movie_matrix[:, target_idx] >= 4)[0]
    common_fans = set(source_fans) & set(target_fans)

    overlap_rate = len(common_fans) / len(source_fans) * 100

    return f"喜欢《{source_movie}》的观众中，{overlap_rate:.0f}% 也喜欢《{target_movie}》"
```

#### 4.2.4 SVD矩阵分解 - 最低可解释性 ⭐

```python
# SVD的潜在因子是抽象的数学概念
# 因子1可能混合了"科幻+大制作+2000年后"等多个概念
# 难以用人类语言解释

# 可能的解释尝试
"《Avatar》和《Inception》在潜在特征空间中位置接近"
# 用户：？？？
```

**提高SVD可解释性的方法：**
```python
def interpret_svd_factors(self, movie_title, top_n=3):
    """尝试解释SVD因子的含义"""
    movie_idx = self.title_to_idx[movie_title]
    movie_factors = self.svd_item_factors[:, movie_idx]

    # 找到该电影最强的几个因子
    top_factor_indices = np.argsort(np.abs(movie_factors))[::-1][:top_n]

    explanations = []
    for factor_idx in top_factor_indices:
        # 找到该因子得分最高的电影
        top_movies_for_factor = np.argsort(self.svd_item_factors[factor_idx])[::-1][:5]
        factor_movies = [self.idx_to_title[i] for i in top_movies_for_factor]

        # 分析这些电影的共同特征
        common_genres = self._find_common_genres(factor_movies)
        explanations.append(f"因子{factor_idx}: 可能代表 {common_genres}")

    return explanations
```

### 4.3 可解释性评估框架

```
                    可解释性
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    透明度           直觉性          可操作性
    (算法是否        (解释是否        (用户能否
     易理解)         符合直觉)        据此行动)
        │               │               │
        ▼               ▼               ▼

    基于内容: ★★★★★   ★★★★★   ★★★★☆
    基于元数据: ★★★★★ ★★★★★   ★★★★★
    Item-CF: ★★★☆☆   ★★★★☆   ★★★☆☆
    SVD: ★☆☆☆☆       ★★☆☆☆   ★☆☆☆☆
```

### 4.4 实际应用中的可解释性展示

```python
# 推荐系统UI应该展示的信息

def get_explainable_recommendation(movie_title, top_n=5):
    """生成带解释的推荐"""
    results = []

    for method in ['content', 'metadata', 'cf']:
        recommendations = get_recommendations(movie_title, method)

        for rec in recommendations[:top_n]:
            result = {
                'title': rec['title'],
                'score': rec['similarity_score'],
                'method': method,
                'explanation': generate_explanation(movie_title, rec['title'], method)
            }
            results.append(result)

    return results

# 输出示例
{
    "title": "Interstellar",
    "score": 0.85,
    "method": "content",
    "explanation": {
        "summary": "因为剧情主题相似",
        "details": [
            "共同关键词: space, survival, future",
            "相似度: 85%"
        ]
    }
}

{
    "title": "Titanic",
    "score": 0.72,
    "method": "metadata",
    "explanation": {
        "summary": "同一导演作品",
        "details": [
            "导演: James Cameron",
            "导演贡献: 60%",
            "类型贡献: 10%"
        ]
    }
}
```

### 4.5 可解释性最佳实践

1. **多层次解释**
   ```
   简短版: "同一导演的作品"
   详细版: "都由Christopher Nolan执导，他擅长复杂叙事和科幻题材"
   技术版: "元数据相似度0.85，其中导演贡献0.6，类型贡献0.2"
   ```

2. **可视化辅助**
   ```
   Avatar ──────┬────── Interstellar
              │
              ├─ 科幻 ●●●●●
              ├─ 冒险 ●●●●○
              └─ 导演 ○○○○○
   ```

3. **对比展示**
   ```
   为什么推荐     为什么不推荐
   ────────────  ────────────
   ✓ 同类型       ✗ 不同年代
   ✓ 高评分       ✗ 时长更长
   ✓ 相似主题
   ```

---

## 5. Web界面实现

### 5.1 Streamlit缓存机制

```python
# app.py: 80-81行
@st.cache_data
def load_and_preprocess_data():
    """加载和预处理数据"""
    # 只在第一次运行时执行，之后直接返回缓存
```

**缓存装饰器的作用：**
```
首次访问                    后续访问
    │                          │
    ▼                          ▼
加载CSV (3秒)              直接返回缓存 (0.01秒)
预处理 (2秒)                    │
计算相似度 (5秒)               ▼
    │                     页面立即显示
    ▼
存入缓存
    │
    ▼
页面显示
```

### 5.2 用户界面交互

```python
# 侧边栏控件
with st.sidebar:
    # 搜索框
    search_query = st.text_input("搜索电影")

    # 电影选择
    selected_movie = st.selectbox("选择一部电影", options=movie_options)

    # 推荐数量滑块
    top_n = st.slider("推荐数量", min_value=5, max_value=20, value=10)

    # 混合权重调节
    w1 = st.slider("内容相似度权重", 0.0, 1.0, 0.3, 0.1)
```

### 5.3 响应式布局

```python
# 使用columns创建多列布局
col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1])

with col1:
    st.markdown(f"### {idx}")  # 排名

with col2:
    st.markdown(f"**{row['title']}**")  # 电影名

with col3:
    st.markdown(f"⭐ {row['vote_average']}/10")  # 评分

with col4:
    st.metric("相似度", f"{score:.1%}")  # 相似度指标
```

---

## 6. 核心代码逐行解析

### 6.1 TF-IDF相似度计算（完整流程）

```python
# Step 1: 创建向量化器
tfidf = TfidfVectorizer(
    stop_words='english',     # 移除 "the", "is" 等常见词
    max_features=5000,        # 限制特征数，防止维度爆炸
    ngram_range=(1, 2)        # 同时使用单词和双词
)
# ngram_range=(1,2) 示例:
# "great movie" → ["great", "movie", "great movie"]

# Step 2: 拟合并转换
tfidf_matrix = tfidf.fit_transform(self.movies['overview'])
# fit: 学习词汇表，计算IDF
# transform: 将文本转为TF-IDF向量
# 结果: 稀疏矩阵 (4803, 5000)

# Step 3: 计算相似度
self.content_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
# linear_kernel(A, B) = A @ B.T
# 对于L2标准化的向量，等价于余弦相似度
# 结果: 密集矩阵 (4803, 4803)
```

### 6.2 推荐结果获取（完整流程）

```python
def content_based_recommend(self, title, top_n=10):
    # Step 1: 获取电影索引
    idx = self.title_to_idx[title]  # "Avatar" → 0

    # Step 2: 获取相似度分数
    sim_scores = list(enumerate(self.content_similarity[idx]))
    # [(0, 1.0), (1, 0.85), (2, 0.72), ...]
    # 第一个元素是自己，相似度为1

    # Step 3: 排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 按相似度降序排列

    # Step 4: 取top_n（排除自己）
    sim_scores = sim_scores[1:top_n + 1]
    # 从索引1开始，跳过自己

    # Step 5: 提取索引和分数
    movie_indices = [i[0] for i in sim_scores]  # [1, 2, 5, ...]
    scores = [i[1] for i in sim_scores]          # [0.85, 0.72, ...]

    # Step 6: 构建结果DataFrame
    recommendations = self.movies.iloc[movie_indices][
        ['title', 'genres_list', 'vote_average', 'overview']
    ].copy()
    recommendations['similarity_score'] = scores

    return recommendations
```

### 6.3 混合推荐权重计算

```python
def hybrid_recommend(self, title, top_n=10, weights=(0.3, 0.4, 0.3)):
    idx = self.title_to_idx[title]

    # 初始化分数向量
    hybrid_scores = np.zeros(n_movies)  # [0, 0, 0, ...]

    # 加权累加
    # weights = (0.3, 0.4, 0.3) → 内容30%, 元数据40%, CF30%

    hybrid_scores += weights[0] * self.content_similarity[idx]
    # [0, 0, 0, ...] + 0.3 * [1.0, 0.85, 0.72, ...] = [0.3, 0.255, 0.216, ...]

    hybrid_scores += weights[1] * self.metadata_similarity[idx]
    # [0.3, 0.255, ...] + 0.4 * [1.0, 0.60, 0.55, ...] = [0.7, 0.495, ...]

    hybrid_scores += weights[2] * self.item_similarity_cf[idx]
    # [0.7, 0.495, ...] + 0.3 * [1.0, 0.70, 0.65, ...] = [1.0, 0.705, ...]

    # 最终分数是三种方法的加权和
```

---

## 总结

### 算法对比表

| 方法 | 原理 | 优点 | 缺点 | 可解释性 |
|------|------|------|------|----------|
| 基于内容 | TF-IDF + 余弦相似度 | 无需用户数据，无冷启动 | 只能推荐相似内容 | ⭐⭐⭐⭐⭐ |
| 基于元数据 | 特征组合 + CountVec | 多维度匹配，直观 | 需要完整元数据 | ⭐⭐⭐⭐⭐ |
| Item-CF | 用户评分模式相似度 | 能发现隐藏关联 | 冷启动问题 | ⭐⭐⭐ |
| User-CF | 用户相似度 | 个性化强 | 稀疏性问题 | ⭐⭐ |
| SVD | 矩阵分解 | 降维，去噪 | 完全黑盒 | ⭐ |
| 混合 | 加权融合 | 取长补短 | 需要调参 | ⭐⭐⭐⭐ |

### 关键技术点

1. **TF-IDF** - 捕捉文本的独特主题词
2. **余弦相似度** - 衡量向量方向的相似性，不受长度影响
3. **特征权重** - 元数据中类型和导演权重更高，符合用户行为
4. **矩阵分解** - 发现用户和电影的潜在特征
5. **混合推荐** - 结合多种方法，提高鲁棒性

### 可解释性核心

- **基于内容**: 共同关键词、相似主题
- **基于元数据**: 同导演、同类型、同演员
- **协同过滤**: 相似用户群体的偏好
- **展示方式**: 多层次解释 + 可视化 + 贡献度分析

---

## 7. SHAP可解释性框架

### 7.1 什么是SHAP？

**SHAP (SHapley Additive exPlanations)** 是基于博弈论的可解释性方法：

```
Shapley值的核心思想:
- 将"预测结果"视为多个特征"合作"的产出
- 公平地分配每个特征对最终预测的"贡献"
- 基于所有可能的特征组合来计算边际贡献
```

**SHAP值的性质：**
1. **局部精确性**: 所有特征的SHAP值之和 = 预测值 - 基准值
2. **一致性**: 如果特征A的贡献始终≥特征B，则SHAP(A) ≥ SHAP(B)
3. **可加性**: 模型预测 = 基准值 + Σ(SHAP值)

### 7.2 SHAP在推荐系统中的应用

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHAP可解释性流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  电影A ──────┐                                                  │
│              │                                                  │
│              ├──► 特征差异/匹配 ──► 替代模型 ──► SHAP解释       │
│              │    (类型、导演、    (梯度提升)   (特征贡献)       │
│  电影B ──────┘     演员、关键词)                                │
│                                                                 │
│  输出示例:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 为什么推荐《Titanic》给喜欢《Avatar》的用户?             │   │
│  │                                                         │   │
│  │ 正向因素:                                               │   │
│  │  +0.25  同一导演 (James Cameron)                        │   │
│  │  +0.08  热度水平相近                                    │   │
│  │                                                         │   │
│  │ 负向因素:                                               │   │
│  │  -0.05  类型不完全匹配 (Sci-Fi vs Drama)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 实现架构

```python
# shap_explainability.py 核心类

class SHAPRecommenderExplainer:
    """
    使用替代模型 + SHAP 解释推荐
    """

    def __init__(self, movies_df, similarity_matrix):
        # 1. 准备电影特征矩阵
        # 2. 训练替代模型模拟相似度计算
        # 3. 使用SHAP解释替代模型

    def _prepare_features(self):
        """
        将电影元数据转换为数值特征:
        - 类型: MultiLabelBinarizer (one-hot)
        - 导演: LabelEncoder
        - 数值特征: vote_average, popularity
        - 计数特征: cast_count, keyword_count
        """

    def train_explainer_model(self):
        """
        训练梯度提升回归模型:
        输入: 电影对的特征差异和匹配度
        输出: 相似度分数
        """
        # 采样电影对
        # 计算特征差异 |f1 - f2| 和特征乘积 f1 * f2
        # 训练 GradientBoostingRegressor
        # 创建 SHAP TreeExplainer

    def explain_pair(self, source_title, target_title):
        """
        解释两部电影的相似度
        返回: SHAP值分解
        """


class FeatureContributionAnalyzer:
    """
    不依赖SHAP的简化版特征贡献分析
    直接计算各特征的匹配度和贡献
    """
```

### 7.4 使用示例

```python
from movie_recommender import MovieRecommenderSystem
from shap_explainability import SHAPRecommenderExplainer, FeatureContributionAnalyzer

# 初始化
recommender = MovieRecommenderSystem('tmdb_5000_movies.csv', 'tmdb_5000_credits.csv')
recommender.build_metadata_based_model()

# 方法1: SHAP解释器
shap_explainer = SHAPRecommenderExplainer(
    recommender.movies,
    recommender.metadata_similarity,
    'metadata'
)
shap_explainer.train_explainer_model()

# 生成SHAP解释报告
report = shap_explainer.generate_explanation_report('Avatar', 'Titanic')
print(report)

# 绘制SHAP特征重要性图
shap_explainer.plot_feature_importance()

# 方法2: 特征贡献分析（无需SHAP依赖）
analyzer = FeatureContributionAnalyzer(recommender.movies)
report = analyzer.generate_contribution_report('Avatar', 'Titanic')
print(report)

# 绘制贡献图
analyzer.plot_contribution('Avatar', 'Titanic')
```

### 7.5 SHAP解释报告示例

```
======================================================================
SHAP 可解释性分析报告
======================================================================

源电影: 《Avatar》
推荐电影: 《Titanic》

【相似度分析】
  实际相似度: 72.3%
  模型预测相似度: 71.8%
  SHAP基准值: 0.1523

【正向影响因素】(提高相似度)
  同一导演的作品 (+0.2534)
  热度水平相近 (+0.0821)
  评分水平相近 (+0.0456)

【负向影响因素】(降低相似度)
  类型 Drama 不匹配 (-0.0523)
  类型 Science Fiction 差异大 (-0.0312)

【解释说明】
  SHAP值表示每个特征对相似度预测的贡献
  正值表示该特征增加相似度，负值表示降低相似度
======================================================================
```

### 7.6 特征贡献可视化

```
【贡献可视化】
  类型   [████████░░░░░░░░░░░░] 40.2%
  导演   [████████████████░░░░] 82.3%
  演员   [░░░░░░░░░░░░░░░░░░░░] 0.0%
  关键词 [██░░░░░░░░░░░░░░░░░░] 12.5%
```

### 7.7 SHAP vs 传统解释方法对比

| 特性 | 传统方法 | SHAP方法 |
|------|----------|----------|
| 数学基础 | 启发式规则 | 博弈论Shapley值 |
| 公平性 | 依赖设计 | 数学保证公平 |
| 一致性 | 不保证 | 保证一致 |
| 可加性 | 不保证 | 保证可加 |
| 计算复杂度 | O(n) | O(2^n) → 近似 |
| 可视化 | 需自定义 | 内置丰富图表 |
| 全局/局部 | 通常局部 | 两者兼顾 |

### 7.8 核心代码解析

```python
def explain_pair(self, source_title, target_title):
    """解释两部电影的相似度"""

    # Step 1: 获取两部电影的特征向量
    f1 = self.feature_matrix[source_idx]  # 例: [1,0,1,0,5.2,...]
    f2 = self.feature_matrix[target_idx]  # 例: [1,1,0,0,4.8,...]

    # Step 2: 计算特征关系
    feature_diff = np.abs(f1 - f2)   # 差异: [0,1,1,0,0.4,...]
    feature_prod = f1 * f2            # 匹配: [1,0,0,0,24.96,...]

    # Step 3: 组合为输入
    X = np.hstack([feature_diff, feature_prod])

    # Step 4: 计算SHAP值
    shap_values = self.explainer.shap_values(X)

    # Step 5: 解释结果
    # shap_values[i] > 0: 第i个特征正向影响相似度
    # shap_values[i] < 0: 第i个特征负向影响相似度
    # sum(shap_values) + base_value ≈ predicted_similarity
```

### 7.9 安装与依赖

```bash
# 安装SHAP
pip install shap>=0.42.0

# 验证安装
python -c "import shap; print(f'SHAP版本: {shap.__version__}')"
```

**注意**: 如果SHAP安装失败，可以使用 `FeatureContributionAnalyzer` 作为替代，它不依赖SHAP包但提供类似的特征贡献分析功能。
