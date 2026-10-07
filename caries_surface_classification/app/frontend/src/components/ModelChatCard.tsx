import React, { useState } from 'react';
import { Bot, Send, Sparkles, MessageSquare, Check, Copy } from 'lucide-react';

interface ModelChatCardProps {
  onResponseReceived?: () => void;
}

export const ModelChatCard: React.FC<ModelChatCardProps> = ({ onResponseReceived }) => {
  const [prompt, setPrompt] = useState<string>('Explain what this dashboard monitors.');
  const [response, setResponse] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const samplePrompts = [
    'Explain what this dashboard monitors.',
    'What happens when a non-panoramic image is uploaded?',
    'How does the Caries Surface Classifier evaluate severity?'
  ];

  const handleSendPrompt = async (textToSend?: string) => {
    const text = textToSend || prompt;
    if (!text.trim()) return;

    setIsLoading(true);
    const start = performance.now();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text })
      });

      const data = await res.json();
      const end = performance.now();
      setLatencyMs(Math.round(end - start));
      setResponse(data.response || 'No response returned.');
      if (onResponseReceived) onResponseReceived();
    } catch (err) {
      console.error('Chat error:', err);
      setResponse('Error communicating with the model endpoint.');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    if (!response) return;
    navigator.clipboard.writeText(response);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Bot className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              AI Model Endpoint & Reasoning Engine
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">
                POST /chat
              </span>
            </h2>
            <p className="text-xs text-slate-400">Interactive Model Query & Diagnostic Assistant</p>
          </div>
        </div>

        {latencyMs !== null && (
          <div className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono text-cyan-300">
            Response Time: <strong>{latencyMs}ms</strong>
          </div>
        )}
      </div>

      {/* Suggested Prompts */}
      <div>
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
          Preset Verification Prompts:
        </span>
        <div className="flex flex-wrap gap-2">
          {samplePrompts.map((p, idx) => (
            <button
              key={idx}
              onClick={() => {
                setPrompt(p);
                handleSendPrompt(p);
              }}
              className="px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-xs text-slate-300 hover:text-cyan-300 transition flex items-center gap-1.5"
            >
              <Sparkles className="h-3 w-3 text-cyan-400" />
              <span>{p}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Input Form */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendPrompt();
        }}
        className="flex gap-2"
      >
        <div className="relative flex-1">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type a prompt to test the model endpoint..."
            className="w-full bg-slate-900/90 border border-slate-700 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 outline-none transition"
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-slate-950 font-bold text-xs uppercase tracking-wider transition shadow-md shadow-emerald-950/40 flex items-center gap-2 disabled:opacity-50"
        >
          {isLoading ? (
            <div className="h-4 w-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
          <span>Send</span>
        </button>
      </form>

      {/* Response Box */}
      <div className="rounded-xl bg-slate-950 border border-slate-800/90 p-4 relative min-h-[140px] flex flex-col">
        <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-900">
          <span className="text-xs font-semibold text-slate-400 flex items-center gap-1.5">
            <MessageSquare className="h-3.5 w-3.5 text-emerald-400" />
            Last Model Response
          </span>

          {response && (
            <button
              onClick={copyToClipboard}
              className="text-xs text-slate-400 hover:text-white flex items-center gap-1 transition p-1 rounded hover:bg-slate-900"
            >
              {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          )}
        </div>

        {isLoading ? (
          <div className="flex-1 flex flex-col items-center justify-center py-6 text-slate-400 space-y-2">
            <div className="h-6 w-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-mono">Querying AI model provider...</span>
          </div>
        ) : response ? (
          <div className="text-xs text-slate-200 leading-relaxed font-sans whitespace-pre-wrap selection:bg-emerald-500/30 selection:text-white">
            {response}
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-xs text-slate-500 font-mono italic">
            Click "Send" or select a preset prompt above to query the model.
          </div>
        )}
      </div>
    </div>
  );
};
