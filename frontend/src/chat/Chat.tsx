import { useMemo } from "react";
import { AssistantRuntimeProvider, ComposerPrimitive, MessagePrimitive, ThreadPrimitive, type ToolCallMessagePartProps, useMessagePartText } from "@assistant-ui/react";
import { ExportedMessageRepository } from "@assistant-ui/core";
import { HttpAgent } from "@ag-ui/client";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, type ConversationDetail } from "../api/client";

type Props = { conversation: ConversationDetail | null; onConversationChanged: (value: ConversationDetail) => void };

function MarkdownText() { return <ReactMarkdown remarkPlugins={[remarkGfm]}>{useMessagePartText().text}</ReactMarkdown>; }
function ToolActivity({ toolName, argsText, result, isError }: ToolCallMessagePartProps) { return <details className="tool-activity" open><summary>{isError ? "工具失败" : "工具调用"} · {toolName}</summary><div className="tool-section"><b>参数</b><pre>{argsText || "正在生成参数…"}</pre></div>{result !== undefined && <div className="tool-section"><b>返回值</b><pre>{JSON.stringify(result, null, 2)}</pre></div>}</details>; }
function AssistantMessage() { return <MessagePrimitive.Root className="message assistant"><div className="role">JobPilot</div><MessagePrimitive.Parts components={{ Text: MarkdownText, tools: { Fallback: ToolActivity } }} /></MessagePrimitive.Root>; }
function UserMessage() { return <MessagePrimitive.Root className="message user"><div className="role">你</div><MessagePrimitive.Parts components={{ Text: MarkdownText }} /></MessagePrimitive.Root>; }
function Thread() { return <ThreadPrimitive.Root className="chat-thread"><ThreadPrimitive.Viewport className="messages"><ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} /></ThreadPrimitive.Viewport><ComposerPrimitive.Root className="composer"><ComposerPrimitive.Input placeholder="输入你的职业问题、JD、简历内容或训练回答…" rows={4} /><div className="composer-actions"><ComposerPrimitive.Cancel>停止</ComposerPrimitive.Cancel><ComposerPrimitive.Send className="primary">发送</ComposerPrimitive.Send></div></ComposerPrimitive.Root></ThreadPrimitive.Root>; }
function RuntimeChat({ conversation, onConversationChanged }: { conversation: ConversationDetail; onConversationChanged: (value: ConversationDetail) => void }) {
  const agent = useMemo(() => new HttpAgent({ url: `/api/ag-ui/conversations/${conversation.id}` }), [conversation.id]);
  const runtime = useAgUiRuntime({ agent, showThinking: false, onError: () => void api.getConversation(conversation.id).then(onConversationChanged), adapters: { history: { async load() { const current = await api.getConversation(conversation.id); return ExportedMessageRepository.fromArray(current.messages.map(message => ({ id: message.id, role: message.role.toLowerCase() as "user" | "assistant", content: [{ type: "text" as const, text: message.content }] }))); }, async append() { /* server persists at run start */ } } } });
  return <AssistantRuntimeProvider runtime={runtime}><Thread /></AssistantRuntimeProvider>;
}
export function Chat({ conversation, onConversationChanged }: Props) { return <div className="chat-page"><header><h1>{conversation?.title || "对话"}</h1><p>与了解你职业背景的长期 Agent 对话。</p></header>{conversation ? <RuntimeChat key={conversation.id} conversation={conversation} onConversationChanged={onConversationChanged} /> : <div className="empty">从左侧新建或选择一段对话开始。</div>}</div>; }
