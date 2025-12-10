import chromadb

persist = "chroma_db"
client = chromadb.PersistentClient(path=persist)

# 1) 列出 collections
cols = client.list_collections()
print("collections:", [c.name for c in cols])

# 2) 逐个检查 count、示例 metadata、向量维度
for col in cols:
    c = client.get_collection(col.name)
    n = c.count()
    print(f"\n[{col.name}] count={n}")

    # 抽样取 1 条，查看 metadata 字段
    if n > 0:
        sample = c.get(limit=1, include=["metadatas", "embeddings", "documents"])
        meta = sample["metadatas"][0]
        emb = sample["embeddings"][0]
        print("sample meta keys:", list(meta.keys()))
        print("embedding dim:", len(emb))
        docs = sample.get("documents")
        if docs and docs[0]:
            print("sample doc head:", docs[0][:120].replace("\n", " "))
        else:
            print("sample doc head: <none>")

# 3) 验证 where 过滤是否可用（按需修改 collection 名和字段值）
text_col = client.get_collection("coffee_text")
# 用一个简单的随机向量或已有查询向量；这里仅示例用全 0 同维度向量
# 更实际的做法：用你的 embedding 模型对 "test" 生成 query 向量
q_emb = [0.0] * len(text_col.get(limit=1, include=["embeddings"])["embeddings"][0])

res = text_col.query(
    query_embeddings=[q_emb],
    n_results=3,
    where={"section_path": "TROUBLESHOOTING"},  # 确认过滤字段可用
    include=["metadatas", "documents"],
)
print("filter test hits:", len(res["ids"][0]))
for m in res["metadatas"][0]:
    print("hit meta:", m)