import React from 'react';
import { Cpu } from 'lucide-react';
import { SystemMetricsResponse } from '../types';

interface CpuUsageCardProps {
  metrics: SystemMetricsResponse | null;
}

export const CpuUsageCard: React.FC<CpuUsageCardProps> = ({ metrics }) => {
  const cpu = metrics?.cpu;
  const uptime = metrics?.uptime;

  return (
    <div className="glass-panel glass-panel-hover rounded-2xl p-5 border border-slate-800 transition-all duration-300">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">CPU & Process Runtime</h3>
            <p className="text-[11px] text-slate-400">Node.js Process Clock</p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-mono font-bold text-amber-400">
            {uptime ? `Uptime ${uptime.formatted}` : '--:--:--'}
          </span>
        </div>
      </div>

      {/* CPU Execution Bar */}
      <div className="mt-3">
        <div className="flex justify-between text-xs text-slate-400 mb-1 font-mono">
          <span>Active Core Utilization</span>
          <span className="text-slate-300">{cpu ? `${cpu.cpuUsagePercentage}%` : '--'}</span>
        </div>
        <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full transition-all duration-700"
            style={{ width: `${Math.min(100, Math.max(3, cpu?.cpuUsagePercentage || 0))}%` }}
          />
        </div>
      </div>

      {/* Detailed CPU Metrics */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">CPU User Time</span>
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-mono font-bold text-amber-300">
              {cpu ? `${cpu.userTimeMs} ms` : '--'}
            </span>
            <span className="text-[10px] text-slate-500 font-mono">
              {cpu ? `${(cpu.userMicroseconds / 1e6).toFixed(2)}s` : ''}
            </span>
          </div>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-0.5">CPU System Time</span>
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-mono font-bold text-orange-300">
              {cpu ? `${cpu.systemTimeMs} ms` : '--'}
            </span>
            <span className="text-[10px] text-slate-500 font-mono">
              {cpu ? `${(cpu.systemMicroseconds / 1e6).toFixed(2)}s` : ''}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
