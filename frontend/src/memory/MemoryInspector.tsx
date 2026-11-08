import { useEffect, useState } from "react";
import { api, type Memory } from "../api/client";

export function MemoryInspector() {
  const [items, setItems] = useState<Memory[]>([]);
  const load = () => api.listMemories().then(setItems).catch(() => setItems([]));
  useEffect(() => { void load(); }, []);
  return <section className="inspector"><header><h1>长期记忆</h1><p>仅保留有证据、会影响未来帮助方式的高密度用户认知。</p></header><div className="cards">{items.map(item => <article className="card" key={item.id}><small>{item.kind} · {item.importance} · {item.status}</small><p>{item.content}</p><button onClick={() => void api.archiveMemory(item.id).then(load)}>归档</button></article>)}{items.length === 0 && <div className="empty">尚无长期记忆。</div>}</div></section>;
}
