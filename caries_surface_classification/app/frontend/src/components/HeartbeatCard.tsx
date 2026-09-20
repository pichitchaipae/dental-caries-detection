import React from 'react';
import { HeartPulse, Clock } from 'lucide-react';
import { HeartbeatResponse } from '../types';

interface HeartbeatCardProps {
  heartbeat: HeartbeatResponse | null;
  lastReceivedAt: Date | null;
  latencyMs: number;
}

export const HeartbeatCard: React.FC<HeartbeatCardProps> = ({
  heartbeat,
  lastReceivedAt,
  latencyMs
}) => {
  const now = new Date().getTime();
  const timeSinceLastPulse = lastReceivedAt ? Math.floor((now - lastReceivedAt.getTime()) / 1000) : 999;
  const isAlive = heartbeat?.status === 'alive' && timeSinceLastPulse < 5;

  return (
    <div className={`glass-panel glass-panel-hover rounded-2xl p-5 border transition-all duration-300 relative overflow-hidden ${
      isAlive ? 'border-emerald-500/40 hover:border-emerald-500/70' : 'border-rose-500/50 hover:border-rose-500/80 animate-glow-red'
    }`}>
      {/* Background ambient glow */}
      <div className={`absolute -right-6 -bottom-6 w-24 h-24 rounded-full blur-2xl opacity-20 pointer-events-none ${
        isAlive ? 'bg-emerald-500' : 'bg-rose-500'
      }`} />

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className={`p-2 rounded-xl border ${
            isAlive ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-rose-500/10 border-rose-500/20 text-rose-400'
          }`}>
            <HeartPulse className={`h-5 w-5 ${isAlive ? 'animate-pulse' : ''}`} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">Heartbeat Monitor</h3>
            <p className="text-[11px] text-slate-400">1-Second Backend Health Pulse</p>
          </div>
        </div>

        {/* Live Status Pill with glowing indicator */}
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-bold tracking-wider uppercase border shadow-sm ${
          isAlive
            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-emerald-500/20'
            : 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-rose-500/20'
        }`}>
          <span className="relative flex h-2 w-2">
            {isAlive && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            )}
            <span className={`relative inline-flex rounded-full h-2 w-2 ${isAlive ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
          </span>
          <span>{isAlive ? 'ALIVE (<5s)' : 'STALE (>=5s)'}</span>
        </div>
      </div>

      {/* Timestamp & Telemetry Details */}
      <div className="mt-4 space-y-2.5">
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
          <span className="text-xs text-slate-400 flex items-center gap-1.5">
            <Clock className="h-3.5 w-3.5 text-cyan-400" />
            Backend ISO Timestamp:
          </span>
          <span className="text-xs font-mono font-semibold text-slate-200">
            {heartbeat?.timestamp ? heartbeat.timestamp.split('T')[1].replace('Z', '') : '--:--:--'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block mb-0.5">Last Pulse Age</span>
            <span className={`text-sm font-mono font-bold ${timeSinceLastPulse < 5 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {lastReceivedAt ? `${timeSinceLastPulse}s ago` : 'Never'}
            </span>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block mb-0.5">Ping Latency</span>
            <span className="text-sm font-mono font-bold text-cyan-400">
              {latencyMs > 0 ? `${latencyMs.toFixed(1)} ms` : '--'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
