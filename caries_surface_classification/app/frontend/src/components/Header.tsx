import React from 'react';
import { Activity, RefreshCw, ShieldCheck, Play, Pause } from 'lucide-react';

interface HeaderProps {
  isOnline: boolean;
  refreshCountdown: number;
  isAutoRefreshActive: boolean;
  onToggleAutoRefresh: () => void;
  onManualRefresh: () => void;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  isOnline,
  refreshCountdown,
  isAutoRefreshActive,
  onToggleAutoRefresh,
  onManualRefresh,
  isLoading
}) => {
  return (
    <header className="glass-panel border-b border-slate-800/80 sticky top-0 z-50 backdrop-blur-md px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Title and Branding */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20 ring-1 ring-white/20">
            <ShieldCheck className="h-6 w-6 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                Dental Caries AI
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  Medical Safety Gate v2.0
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400">
              Panoramic OPG Validation Gate & Real-time Node.js Process Telemetry
            </p>
          </div>
        </div>

        {/* Controls & Server Status */}
        <div className="flex items-center gap-3">
          {/* Server Connectivity Status Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <div className={`h-2.5 w-2.5 rounded-full ${isOnline ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]' : 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'}`} />
            <span className="text-xs font-mono font-medium text-slate-300">
              {isOnline ? 'SERVER ONLINE' : 'DISCONNECTED'}
            </span>
          </div>

          {/* Auto Refresh Countdown & Controls */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <Activity className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-xs font-mono text-slate-400">
              Auto: <span className="text-cyan-300 font-semibold">{isAutoRefreshActive ? `${refreshCountdown}s` : 'PAUSED'}</span>
            </span>
            <button
              onClick={onToggleAutoRefresh}
              title={isAutoRefreshActive ? 'Pause Auto Refresh' : 'Resume Auto Refresh'}
              className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition"
            >
              {isAutoRefreshActive ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3 text-emerald-400" />}
            </button>
          </div>

          {/* Manual Refresh Button */}
          <button
            onClick={onManualRefresh}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 font-medium text-xs transition active:scale-95 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh Now</span>
          </button>
        </div>
      </div>
    </header>
  );
};
