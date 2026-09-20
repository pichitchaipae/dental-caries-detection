import React from 'react';
import { HardDrive } from 'lucide-react';
import { SystemMetricsResponse } from '../types';

interface MemoryUsageCardProps {
  metrics: SystemMetricsResponse | null;
}

export const MemoryUsageCard: React.FC<MemoryUsageCardProps> = ({ metrics }) => {
  const mem = metrics?.memory;
  const heapPercent = mem?.heapUsagePercentage || 0;

  return (
    <div className="glass-panel glass-panel-hover rounded-2xl p-5 border border-slate-800 transition-all duration-300">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
            <HardDrive className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">Memory Allocation</h3>
            <p className="text-[11px] text-slate-400">V8 Process Memory Pool</p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-mono font-bold text-purple-400">
            {mem ? `${heapPercent}% Heap` : '--'}
          </span>
        </div>
      </div>

      {/* Heap Usage Visual Bar */}
      <div className="mt-3">
        <div className="flex justify-between text-xs text-slate-400 mb-1 font-mono">
          <span>Heap Utilization</span>
          <span className="text-slate-300">{mem ? `${mem.heapUsed.formatted} / ${mem.heapTotal.formatted}` : '--'}</span>
        </div>
        <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full transition-all duration-700"
            style={{ width: `${Math.min(100, Math.max(2, heapPercent))}%` }}
          />
        </div>
      </div>

      {/* Detailed Memory Tiles */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">Heap Used</span>
          <span className="text-xs font-mono font-bold text-purple-300">
            {mem?.heapUsed.formatted || '--'}
          </span>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">Heap Total</span>
          <span className="text-xs font-mono font-bold text-indigo-300">
            {mem?.heapTotal.formatted || '--'}
          </span>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">RSS</span>
          <span className="text-xs font-mono font-bold text-cyan-300">
            {mem?.rss.formatted || '--'}
          </span>
        </div>
      </div>
    </div>
  );
};
