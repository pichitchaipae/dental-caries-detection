import React from 'react';
import { ArrowUpDown } from 'lucide-react';
import { SystemMetricsResponse } from '../types';

interface RequestStatsCardProps {
  metrics: SystemMetricsResponse | null;
}

export const RequestStatsCard: React.FC<RequestStatsCardProps> = ({ metrics }) => {
  const traffic = metrics?.traffic;

  return (
    <div className="glass-panel glass-panel-hover rounded-2xl p-5 border border-slate-800 transition-all duration-300">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400">
            <ArrowUpDown className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">Network & Traffic</h3>
            <p className="text-[11px] text-slate-400">HTTP Ingress & Latency</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono font-semibold">
          <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-ping" />
          <span>Active: {traffic?.activeRequests || 0}</span>
        </div>
      </div>

      {/* Latency & Request Counters */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">Total Requests</span>
          <span className="text-sm font-mono font-bold text-blue-300">
            {traffic?.totalRequests !== undefined ? traffic.totalRequests : '--'}
          </span>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">Last Latency</span>
          <span className="text-sm font-mono font-bold text-cyan-300">
            {traffic?.lastLatencyMs !== undefined ? `${traffic.lastLatencyMs} ms` : '--'}
          </span>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">Avg Latency</span>
          <span className="text-sm font-mono font-bold text-emerald-300">
            {traffic?.averageLatencyMs !== undefined ? `${traffic.averageLatencyMs} ms` : '--'}
          </span>
        </div>
      </div>
    </div>
  );
};
