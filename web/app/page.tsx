"use client";

import { Fragment, useState } from "react";

type Chunk = {
  chunk_id: string;
  doc_name: string;
  company: string;
  doc_type: string;
  fiscal_year: number | null;
  pages: number[];
  section: string | null;
  chunk_type: string;
  score: number;
  text: string;
  cited: boolean;
};

type Result = {
  question: string;
  answer: string;
  status: "answered" | "refused" | "refused_invalid_citations";
  model: string;
  fallback_reason: string | null;
  citations: Omit<Chunk, "section" | "chunk_type" | "score" | "text" | "cited">[];
  chunks: Chunk[];
  attempts: number;
  latency_ms: { retrieval: number; generation: number };
};

const EXAMPLES = [
  "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
  "What are the geographies that Pepsico primarily operates in as of FY2022?",
  "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
  "What is Best Buy's FY2023 inventory amount in USD millions?",
];

const CITE_RE = /\[([^\[\]]*::p\d{4}::\d{2}[^\[\]]*)\]|【([^【】]*::p\d{4}::\d{2}[^【】]*)】/g;
const ID_RE = /[A-Za-z0-9_.&-]+::p\d{4}::\d{2}/g;

function pages(p: number[]) {
  return p.length === 1 ? `p. ${p[0]}` : `pp. ${p[0]}–${p[p.length - 1]}`;
}

function label(c: { company: string; doc_type: string; fiscal_year: number | null }) {
  return `${c.company} ${c.doc_type}${c.fiscal_year ? ` FY${c.fiscal_year}` : ""}`;
}

/** Render the answer with each [chunk_id] citation replaced by a numbered link to its chunk. */
function AnswerText({ text, order }: { text: string; order: Map<string, number> }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE_RE)) {
    parts.push(text.slice(last, m.index));
    const ids = (m[1] ?? m[2]).match(ID_RE) ?? [];
    parts.push(
      <Fragment key={m.index}>
        {ids.map((id) => (
          <a key={id} className="cite" href={`#${encodeURIComponent(id)}`} title={id}>
            {order.get(id) ?? "?"}
          </a>
        ))}
      </Fragment>,
    );
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(text.slice(last));
  return <div className="answer">{parts}</div>;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function ask(q: string) {
    if (!q.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      setResult(body as Result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const order = new Map<string, number>();
  result?.chunks.forEach((c, i) => order.set(c.chunk_id, i + 1));
  const refused = result && result.status !== "answered";

  return (
    <main>
      <h1>Footnote</h1>
      <p className="sub">
        Cited answers from the 84 SEC filings in{" "}
        <a href="https://github.com/patronus-ai/financebench">FinanceBench</a>. Every claim links to the chunk it
        came from. <a href="https://github.com/vtalla-hq/footnote">Eval results and code</a>.
      </p>

      <form onSubmit={(e) => { e.preventDefault(); ask(question); }}>
        <textarea
          value={question}
          maxLength={600}
          placeholder="Ask about a company in the set, e.g. 3M, Amazon, AMD, Best Buy, PepsiCo, Verizon…"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(question); }}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Answering…" : "Ask"}
        </button>
      </form>
      <div className="examples">
        {EXAMPLES.map((q) => (
          <button key={q} type="button" disabled={loading} onClick={() => { setQuestion(q); ask(q); }}>
            {q.length > 62 ? q.slice(0, 60) + "…" : q}
          </button>
        ))}
      </div>

      {error && <div className="card error">Request failed: {error}</div>}

      {result && (
        <>
          <div className="card">
            {refused ? (
              <div className="answer">
                <strong>Not answered.</strong>{" "}
                {result.answer.replace(/^INSUFFICIENT_CONTEXT:\s*/i, "")}
              </div>
            ) : (
              <AnswerText text={result.answer} order={order} />
            )}
            {result.fallback_reason && (
              <div className="notice">
                Answered by {result.model}: the primary model (gpt-oss-120b, used in the eval) was rate-limited.
              </div>
            )}
            <div className="meta">
              {result.model} · retrieval {result.latency_ms.retrieval} ms · generation{" "}
              {(result.latency_ms.generation / 1000).toFixed(1)} s
              {result.attempts > 1 && " · regenerated once after a citation check failed"}
            </div>
          </div>

          {result.citations.length > 0 && (
            <>
              <h2>Citations</h2>
              <ul className="sources">
                {result.citations.map((c) => (
                  <li key={c.chunk_id}>
                    <a className="cite" href={`#${encodeURIComponent(c.chunk_id)}`}>{order.get(c.chunk_id)}</a>{" "}
                    {label(c)}, {pages(c.pages)} <span className="mono">({c.doc_name})</span>
                  </li>
                ))}
              </ul>
            </>
          )}

          <h2>Retrieved chunks (top {result.chunks.length}, dense bge-small)</h2>
          {result.chunks.map((c, i) => (
            <details className="chunk" key={c.chunk_id} id={encodeURIComponent(c.chunk_id)} open={c.cited}>
              <summary>
                <strong>{i + 1}.</strong> {label(c)}, {pages(c.pages)}
                {c.section ? ` · ${c.section.slice(0, 70)}` : ""}
                {c.cited && <span className="tag">cited</span>}
                {c.chunk_type === "table" && <span className="tag">table</span>}
                <span className="tag">{c.score.toFixed(3)}</span>
              </summary>
              <pre>{c.text}</pre>
            </details>
          ))}
        </>
      )}

      <footer>
        Retrieval: dense bge-small over 256-token chunks (pgvector), top 5. Generation: gpt-oss-120b on Groq,
        constrained to the retrieved context and citation-checked; gpt-oss-20b if 120b is rate-limited. Dataset:
        FinanceBench (CC-BY-NC 4.0), non-commercial use only.
      </footer>
    </main>
  );
}
