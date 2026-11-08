import { useEffect, useRef, useState } from "react";
import { api, type Document } from "../api/client";

export function KnowledgeInspector() {
  const [items, setItems] = useState<Document[]>([]); const fileInput = useRef<HTMLInputElement>(null);
  const load = () => api.listDocuments().then(setItems).catch(() => setItems([]));
  useEffect(() => { void load(); }, []);
  const upload = async (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase(); const format = ext === "md" ? "MARKDOWN" : ext === "tex" ? "LATEX" : ext === "txt" ? "TEXT" : null;
    if (!format) return; await api.uploadDocument({ title: file.name, filename: file.name, format, content: await file.text() }); await load();
  };
  const remove = async (item: Document) => { if (window.confirm(`删除后会移除“${item.title}”及其 Section、Chunk、Embedding。此操作不可撤销。`)) { await api.deleteDocument(item.id); await load(); } };
  return <section className="inspector"><header><h1>资料库</h1><p>职业原始资料将被解析、分块与检索；不会自动写入长期记忆。</p><input hidden ref={fileInput} type="file" accept=".md,.txt,.tex" onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file); }} /><button className="primary" onClick={() => fileInput.current?.click()}>上传资料</button></header><div className="cards">{items.map(item => <article className="card" key={item.id}><small>{item.format} · {item.source_role} · {item.status}</small><p>{item.title}</p>{item.status === "READY" && <button onClick={() => void api.archiveDocument(item.id).then(load)}>归档</button>}<button className="danger" onClick={() => void remove(item)}>删除</button></article>)}{items.length === 0 && <div className="empty">上传 .md、.txt 或 .tex 职业资料。</div>}</div></section>;
}
