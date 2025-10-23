"use client";

import { useEffect, useRef, useState } from "react";
import { Inter } from "next/font/google";
const inter = Inter({ subsets: ["latin"] });

type Citation = { doc_id: string; section?: string; snippet?: string };
type ChatItem =
  | { role: "user"; text: string }
  | { role: "assistant"; answer: string; citations?: Citation[] };

// Rotating placeholder tips
const TIPS = [
  "How long have the symptoms been going on?",
  "Did the patient describe any triggers?",
  "What tests were ordered?",
  "Any concerns about side effects?",
  "Has it worsened recently?",
  "Any recent travel or exposures?",
  "What follow-up was recommended?",
  "What medications are being used?",
];

export default function Home() {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState<ChatItem[]>([]);
  const [session, setSession] = useState<string>("");
  const [ph, setPh] = useState<string>(TIPS[0]);
  const scrollerRef = useRef<HTMLDivElement>(null);

  // Session: client-only (hydration-safe)
  useEffect(() => {
    const key = "ds_session_id";
    let id = typeof window !== "undefined" ? window.localStorage.getItem(key) : null;
    if (!id) {
      const makeId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? () => crypto.randomUUID()
          : () => Math.random().toString(36).slice(2);
      id = makeId();
      if (typeof window !== "undefined") window.localStorage.setItem(key, id);
    }
    setSession(id!);
  }, []);

  // Rotate placeholder every 3.5s
  useEffect(() => {
    let i = 0;
    const id = setInterval(() => {
      i = (i + 1) % TIPS.length;
      setPh(TIPS[i]);
    }, 3500);
    return () => clearInterval(id);
  }, []);

  // Auto-scroll to newest
  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [items, loading]);

  const ensureSession = () => {
    if (session) return session;
    const makeId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? () => crypto.randomUUID()
        : () => Math.random().toString(36).slice(2);
    const id = makeId();
    if (typeof window !== "undefined") window.localStorage.setItem("ds_session_id", id);
    setSession(id);
    return id;
  };

  const ask = async (text?: string) => {
    const curr = (text ?? q).trim();
    if (!curr) return;

    // clear input immediately
    if (!text) setQ("");

    const sid = ensureSession();
    const Backend = process.env.NEXT_PUBLIC_BACKEND_URL!;
    setItems((m) => [...m, { role: "user", text: curr }]);
    setLoading(true);
    try {
      const res = await fetch(`${Backend}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: curr, session_id: sid }),
      });
      const data = await res.json();
      setItems((m) => [
        ...m,
        { role: "assistant", answer: data.answer, citations: data.citations },
      ]);
    } catch (e: any) {
      setItems((m) => [
        ...m,
        { role: "assistant", answer: `Error contacting backend: ${e?.message ?? "unknown error"}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const newChat = () => {
    const makeId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? () => crypto.randomUUID()
        : () => Math.random().toString(36).slice(2);
    const id = makeId();
    if (typeof window !== "undefined") window.localStorage.setItem("ds_session_id", id);
    setSession(id);
    setItems([]);
    setQ("");
  };

  return (
    <main className={`${inter.className} min-h-screen bg-gray-50 text-gray-900 flex flex-col`}>
      {/* Header */}
      <header className="sticky top-0 bg-white shadow-sm border-b border-gray-200 z-10">
        <div className="max-w-5xl mx-auto px-6 py-3 flex justify-between items-center">
          <h1 className="text-xl font-semibold tracking-tight text-gray-900">
            MedQuery Chat
          </h1>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-gray-500" suppressHydrationWarning>
              session: {session ? `${session.slice(0, 8)}…` : "—"}
            </span>
            <button
              onClick={newChat}
              className="px-3 py-1.5 rounded-md border border-gray-300 hover:bg-gray-100 transition"
            >
              New chat
            </button>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <div
        ref={scrollerRef}
        className="flex-1 max-w-3xl w-full mx-auto px-4 py-8 space-y-6 overflow-y-auto"
      >
        {items.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="bg-blue-600 text-white px-4 py-2 rounded-2xl shadow-sm max-w-[80%]">
                {m.text}
              </div>
            </div>
          ) : (
            <div
              key={i}
              className="bg-white border border-gray-200 shadow-sm rounded-2xl p-5 max-w-[90%]"
            >
              <p className="whitespace-pre-wrap leading-relaxed">{m.answer}</p>

              {!!m.citations?.length && (
                <details className="mt-4">
                  <summary className="text-sm text-blue-700 hover:underline cursor-pointer select-none">
                    Show citations
                  </summary>
                  <div className="mt-2 text-sm text-gray-600">
                    <ul className="space-y-1 list-disc ml-5">
                      {m.citations.map((c, j) => (
                        <li key={j}>
                          {c.doc_id === "soap" ? `SOAP ${c.section ?? ""}` : "Transcript"} —{" "}
                          <span className="italic text-gray-500">{c.snippet}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </details>
              )}
            </div>
          )
        )}

        {loading && (
          <div className="text-center text-sm text-gray-400 animate-pulse">Thinking…</div>
        )}
      </div>

      {/* Helper text */}
      <p className="max-w-3xl mx-auto px-4 pb-1 text-xs text-gray-500">
        What would you like to know about the patient visit?
      </p>

      {/* Input bar */}
      <div className="sticky bottom-0 bg-white border-t border-gray-200">
        <div className="max-w-3xl mx-auto px-4 py-4 flex gap-3">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !loading && ask()}
            placeholder={ph}
            className="flex-1 border border-gray-300 rounded-full px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
          />
          <button
            onClick={() => ask()}
            disabled={loading || !q.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed text-white font-medium px-5 py-2 rounded-full shadow-sm transition"
          >
            Ask
          </button>
        </div>
      </div>
    </main>
  );
}