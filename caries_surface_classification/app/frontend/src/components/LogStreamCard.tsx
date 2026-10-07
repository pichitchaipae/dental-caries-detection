import React, { useState } from 'react';
import { Terminal, Search, RefreshCw } from 'lucide-react';
import { LogEntry } from '../types';

interface LogStreamCardProps {
  logs: LogEntry[];
  onRefreshLogs: () => void;
  isLoading: boolean;
}

export const LogStreamCard: React.FC<LogStreamCardProps> = ({ logs, onRefreshLogs, isLoading }) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedMethod, setSelectedMethod] = useState<string>('ALL');

  const filteredLogs = logs.filter((log) => {
    const matchesSearch =
      log.path.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.status.toString().includes(searchTerm);

    const matchesMethod = selectedMethod === 'ALL' || log.method.toUpperCase() === selectedMethod;
    return matchesSearch && matchesMethod;
  });

  const getStatusBadgeClass = (status: number) => {
    if (status >= 200 && status < 300) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    if (status >= 400 && status < 500) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  };

  const getMethodBadgeClass = (method: string) => {
    switch (method.toUpperCase()) {
      case 'GET':
        return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
      case 'POST':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
      case 'PUT':
      case 'PATCH':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'DELETE':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
      default:
        return 'bg-slate-700/50 text-slate-300 border-slate-600';
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Terminal className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              System & Ingress Audit Stream
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-mono">
                LAST {logs.length} EVENTS
              </span>
            </h2>
            <p className="text-xs text-slate-400">Real-time HTTP Request Ring Buffer (GET /logs)</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={onRefreshLogs}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-mono text-slate-300 hover:text-white transition flex items-center gap-1.5"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Fetch Logs</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-2">
        <div className="relative flex-1 w-full">
          <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Filter logs by route, status code, or message..."
            className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Method filter chips */}
        <div className="flex items-center gap-1 w-full sm:w-auto">
          {['ALL', 'GET', 'POST'].map((m) => (
            <button
              key={m}
              onClick={() => setSelectedMethod(m)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition ${
                selectedMethod === m
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Log Entries Table / Stream */}
      <div className="rounded-xl border border-slate-800/90 bg-slate-950 overflow-hidden">
        <div className="max-h-72 overflow-y-auto divide-y divide-slate-900 text-xs font-mono">
          {filteredLogs.length > 0 ? (
            filteredLogs.map((log) => (
              <div
                key={log.id}
                className="p-2.5 hover:bg-slate-900/60 transition flex flex-col sm:flex-row sm:items-center justify-between gap-2"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getMethodBadgeClass(log.method)}`}>
                    {log.method}
                  </span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getStatusBadgeClass(log.status)}`}>
                    {log.status}
                  </span>
                  <span className="text-slate-200 truncate font-semibold" title={log.path}>
                    {log.path}
                  </span>
                </div>

                <div className="flex items-center gap-3 text-slate-500 text-[11px] shrink-0 self-end sm:self-auto">
                  <span className="text-cyan-400">{log.durationMs.toFixed(1)}ms</span>
                  <span>{log.timestamp.split('T')[1].replace('Z', '')}</span>
                </div>
              </div>
            ))
          ) : (
            <div className="py-8 text-center text-slate-500 italic">
              No matching logs found in ring buffer.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
