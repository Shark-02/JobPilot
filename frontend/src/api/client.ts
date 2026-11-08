export type Conversation = { id: string; title: string | null; created_at: string; updated_at: string };
export type Message = { id: string; conversation_id: string; role: "USER" | "ASSISTANT"; content: string; metadata: Record<string, unknown>; created_at: string };
export type ConversationDetail = Conversation & { messages: Message[] };
export type Document = { id: string; title: string; filename: string | null; format: string; source_role: string; status: string; sha256: string; metadata: Record<string, unknown>; created_at: string; updated_at: string };
export type Memory = { id: string; kind: string; content: string; attributes: Record<string, unknown>; importance: string; status: string; valid_from: string | null; valid_to: string | null; created_at: string; updated_at: string };
export type Run = { turn_id: string; status: string; events_url: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) }, ...init });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Request failed (${response.status})`);
  }
  return response.status === 204 ? (undefined as T) : response.json() as Promise<T>;
}

export const api = {
  listConversations: () => request<Conversation[]>("/api/conversations"),
  createConversation: (title?: string) => request<Conversation>("/api/conversations", { method: "POST", body: JSON.stringify({ title }) }),
  getConversation: (id: string) => request<ConversationDetail>(`/api/conversations/${id}`),
  run: (id: string, content: string, attachment_document_ids: string[]) => request<Run>(`/api/conversations/${id}/runs`, { method: "POST", body: JSON.stringify({ content, attachment_document_ids }) }),
  cancel: (turnId: string) => request<Run>(`/api/turns/${turnId}/cancel`, { method: "POST" }),
  listDocuments: () => request<Document[]>("/api/documents"),
  uploadDocument: (data: { title: string; content: string; format: string; filename?: string }) => request<Document>("/api/documents", { method: "POST", body: JSON.stringify(data) }),
  archiveDocument: (id: string) => request<Document>(`/api/documents/${id}/archive`, { method: "POST" }),
  deleteDocument: (id: string) => request<{ deleted: boolean; document_id: string; deleted_sections: number; deleted_chunks: number }>(`/api/documents/${id}`, { method: "DELETE" }),
  listMemories: () => request<Memory[]>("/api/memories"),
  archiveMemory: (id: string) => request<Memory>(`/api/memories/${id}/archive`, { method: "POST" }),
  deleteMemory: (id: string) => request<void>(`/api/memories/${id}`, { method: "DELETE" }),
  getTurn: (id: string) => request<Record<string, unknown>>(`/api/debug/turns/${id}`),
  modelCalls: (id: string) => request<Array<Record<string, unknown>>>(`/api/debug/turns/${id}/model-calls`),
  toolCalls: (id: string) => request<Array<Record<string, unknown>>>(`/api/debug/turns/${id}/tool-calls`),
  egress: (id: string) => request<Record<string, unknown>>(`/api/debug/model-calls/${id}/egress`),
  listObservabilityTurns: () => request<Array<Record<string, unknown>>>("/api/observability/turns"),
  getObservabilityTurn: (id: string) => request<Record<string, unknown>>(`/api/observability/turns/${id}`),
  observations: (id: string) => request<Array<Record<string, unknown>>>(`/api/observability/turns/${id}/observations`),
  observation: (id: string) => request<Record<string, unknown>>(`/api/observability/observations/${id}`)
};
