import { useEffect, useState } from "react";
import { api, type Conversation, type ConversationDetail } from "../api/client";
import { Chat } from "../chat/Chat";
import { MemoryInspector } from "../memory/MemoryInspector";
import { KnowledgeInspector } from "../knowledge/KnowledgeInspector";
import { DebugInspector } from "../debug/DebugInspector";

type Page = "chat" | "memory" | "knowledge" | "debug";

export function App() {
  const [page, setPage] = useState<Page>("chat");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [current, setCurrent] = useState<ConversationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadConversations = async () => {
    try { setConversations(await api.listConversations()); } catch (reason) { setError(String(reason)); }
  };
  useEffect(() => { void loadConversations(); }, []);

  const select = async (id: string) => {
    try { setCurrent(await api.getConversation(id)); setPage("chat"); } catch (reason) { setError(String(reason)); }
  };
  const create = async () => {
    try {
      const item = await api.createConversation();
      await loadConversations();
      await select(item.id);
    } catch (reason) { setError(String(reason)); }
  };

  return <main className="shell">
    <aside className="sidebar">
      <div className="brand">JobPilot</div>
      <button className="primary" onClick={() => void create()}>新建对话</button>
      <nav aria-label="Main navigation">
        {([ ["chat", "对话"], ["memory", "长期记忆"], ["knowledge", "资料库"], ["debug", "Observability"] ] as const).map(([key, label]) =>
          <button key={key} className={page === key ? "nav active" : "nav"} onClick={() => setPage(key)}>{label}</button>
        )}
      </nav>
      <div className="conversation-list">
        <h2>对话</h2>
        {conversations.map(item => <button className={current?.id === item.id ? "conversation active" : "conversation"} key={item.id} onClick={() => void select(item.id)}>
          {item.title || "未命名对话"}
        </button>)}
      </div>
    </aside>
    <section className="content">
      {error && <div className="error" role="alert">{error}<button onClick={() => setError(null)}>关闭</button></div>}
      {page === "chat" && <Chat conversation={current} onConversationChanged={setCurrent} />}
      {page === "memory" && <MemoryInspector />}
      {page === "knowledge" && <KnowledgeInspector />}
      {page === "debug" && <DebugInspector />}
    </section>
  </main>;
}
