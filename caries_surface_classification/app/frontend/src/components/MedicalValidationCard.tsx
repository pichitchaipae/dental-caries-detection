import React, { useState } from 'react';
import { ShieldCheck, ShieldAlert, UploadCloud, CheckCircle2, XCircle, Lock, Unlock, Sparkles } from 'lucide-react';
import { ValidationResult, CariesClassificationResult } from '../types';

interface MedicalValidationCardProps {
  onValidated?: (res: ValidationResult) => void;
}

export const MedicalValidationCard: React.FC<MedicalValidationCardProps> = ({ onValidated }) => {
  const [selectedPreset, setSelectedPreset] = useState<string>('panoramic_opg');
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [cariesResult, setCariesResult] = useState<CariesClassificationResult | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedBase64, setUploadedBase64] = useState<string | null>(null);

  const presets = [
    { key: 'panoramic_opg', label: '1. Valid Panoramic X-Ray (OPG)', type: 'OPG', valid: true },
    { key: 'selfie', label: '2. Selfie Image', type: 'OOD Photo', valid: false },
    { key: 'screenshot', label: '3. Desktop Screenshot', type: 'OOD UI', valid: false },
    { key: 'intraoral', label: '4. Intraoral Photo', type: 'OOD Macro', valid: false },
    { key: 'bitewing', label: '5. Bitewing X-Ray', type: 'Non-Panoramic', valid: false }
  ];

  const runValidation = async (presetKey?: string, fileData?: { base64: string; name: string; mimeType: string }) => {
    setIsValidating(true);
    setCariesResult(null);

    try {
      let body: any = {};
      if (fileData) {
        body = {
          base64: fileData.base64,
          filename: fileData.name,
          mimeType: fileData.mimeType
        };
      } else {
        body = { presetKey: presetKey || selectedPreset };
      }

      const res = await fetch('/validate-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const data: ValidationResult = await res.json();
      setValidationResult(data);
      if (onValidated) onValidated(data);
    } catch (err) {
      console.error('Validation error:', err);
      setValidationResult({
        accepted: false,
        imageType: 'unknown',
        confidence: 0,
        coverageRatio: 0,
        maskAreaRatio: 0,
        validMask: false,
        reason: 'Network or validation server error.'
      });
    } finally {
      setIsValidating(false);
    }
  };

  const runCariesInference = async () => {
    if (!validationResult || !validationResult.accepted) {
      alert('Medical Safety Gate Violation: Caries classification is strictly blocked for rejected inputs.');
      return;
    }

    setIsClassifying(true);
    try {
      const res = await fetch('/classify-caries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          presetKey: uploadedFileName ? null : selectedPreset,
          base64: uploadedBase64,
          filename: uploadedFileName
        })
      });

      const data: CariesClassificationResult = await res.json();
      setCariesResult(data);
    } catch (err) {
      console.error('Caries classification error:', err);
    } finally {
      setIsClassifying(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFileName(file.name);
    setSelectedPreset('');

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      setUploadedBase64(base64);
      runValidation(undefined, {
        base64,
        name: file.name,
        mimeType: file.type || 'image/jpeg'
      });
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-xl border border-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              Medical Input Validation Gate
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-mono">
                SAFETY LEVEL 1
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Strict 7-Step Panoramic Radiograph (OPG) Gatekeeper & Out-of-Distribution (OOD) Filter
            </p>
          </div>
        </div>

        {/* Safety Gate Status Banner */}
        {validationResult && (
          <div className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold uppercase tracking-wider border shadow-md transition-all ${
            validationResult.accepted
              ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300 shadow-emerald-500/10'
              : 'bg-rose-500/15 border-rose-500/40 text-rose-300 shadow-rose-500/10'
          }`}>
            {validationResult.accepted ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                <span>ACCEPTED (PANORAMIC X-RAY)</span>
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 text-rose-400" />
                <span>REJECTED (OOD INPUT)</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Preset & Input Selectors */}
      <div className="mt-5 grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Input Selection & Dropzone */}
        <div className="lg:col-span-5 space-y-4">
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              1. Select Test Preset or Upload Radiograph
            </label>
            <div className="grid grid-cols-1 gap-2">
              {presets.map((preset) => (
                <button
                  key={preset.key}
                  onClick={() => {
                    setSelectedPreset(preset.key);
                    setUploadedFileName(null);
                    setUploadedBase64(null);
                    runValidation(preset.key);
                  }}
                  className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-medium border flex items-center justify-between transition-all ${
                    selectedPreset === preset.key && !uploadedFileName
                      ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-200 shadow-sm'
                      : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800/60'
                  }`}
                >
                  <span className="font-semibold">{preset.label}</span>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                    preset.valid ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                  }`}>
                    {preset.valid ? 'EXPECT: PASS' : 'EXPECT: REJECT'}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Custom File Upload Dropzone */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              Or Custom File Upload (PNG / JPEG Only)
            </label>
            <label className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 bg-slate-900/40 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition group">
              <UploadCloud className="h-6 w-6 text-slate-400 group-hover:text-cyan-400 transition mb-1" />
              <span className="text-xs text-slate-300 font-medium">
                {uploadedFileName ? uploadedFileName : 'Click to upload image'}
              </span>
              <span className="text-[10px] text-slate-500 mt-0.5">Accepts PNG and JPEG radiographs</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={handleFileUpload}
              />
            </label>
          </div>

          {/* Validation Trigger Button */}
          <button
            onClick={() => runValidation()}
            disabled={isValidating}
            className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-600 to-emerald-600 hover:from-cyan-500 hover:to-emerald-500 text-slate-950 font-bold text-xs uppercase tracking-wider transition shadow-md shadow-cyan-900/30 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isValidating ? (
              <>
                <div className="h-4 w-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Evaluating 7-Step Safety Gate...</span>
              </>
            ) : (
              <>
                <ShieldCheck className="h-4 w-4" />
                <span>Run Medical Gate Validation</span>
              </>
            )}
          </button>
        </div>

        {/* Right: Validation Telemetry & Metrics Gate */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Safety Gate Evaluation Metrics
              </span>
              <span className="text-xs font-mono text-slate-500">
                Image Class: <strong className="text-slate-200">{validationResult?.imageType || 'Pending'}</strong>
              </span>
            </div>

            {/* Metrics Gauges */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Detection Confidence */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                  <span>Confidence</span>
                  <span className="text-slate-500 font-mono">≥ 0.80</span>
                </div>
                <div className="text-lg font-bold font-mono text-white">
                  {validationResult ? (validationResult.confidence * 100).toFixed(0) + '%' : '--'}
                </div>
                {/* Progress bar */}
                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      (validationResult?.confidence || 0) >= 0.80 ? 'bg-emerald-400' : 'bg-rose-500'
                    }`}
                    style={{ width: `${Math.min(100, (validationResult?.confidence || 0) * 100)}%` }}
                  />
                </div>
              </div>

              {/* Coverage Ratio */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                  <span>Coverage Ratio</span>
                  <span className="text-slate-500 font-mono">≥ 0.10</span>
                </div>
                <div className="text-lg font-bold font-mono text-white">
                  {validationResult ? (validationResult.coverageRatio * 100).toFixed(0) + '%' : '--'}
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      (validationResult?.coverageRatio || 0) >= 0.10 ? 'bg-emerald-400' : 'bg-rose-500'
                    }`}
                    style={{ width: `${Math.min(100, ((validationResult?.coverageRatio || 0) / 0.5) * 100)}%` }}
                  />
                </div>
              </div>

              {/* Segmentation Mask Area */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                  <span>Mask Area Ratio</span>
                  <span className="text-slate-500 font-mono">≥ 0.15</span>
                </div>
                <div className="text-lg font-bold font-mono text-white">
                  {validationResult ? (validationResult.maskAreaRatio * 100).toFixed(0) + '%' : '--'}
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      (validationResult?.maskAreaRatio || 0) >= 0.15 ? 'bg-emerald-400' : 'bg-rose-500'
                    }`}
                    style={{ width: `${Math.min(100, ((validationResult?.maskAreaRatio || 0) / 0.5) * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Rejection / Acceptance Reason Box */}
            {validationResult && (
              <div className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                validationResult.accepted
                  ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-200'
                  : 'bg-rose-950/40 border-rose-500/30 text-rose-200'
              }`}>
                <div className="font-bold flex items-center gap-1.5 mb-1">
                  {validationResult.accepted ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-emerald-400 inline" />
                      <span>GATE STATUS: AUTHORIZED FOR CARIES CLASSIFICATION</span>
                    </>
                  ) : (
                    <>
                      <ShieldAlert className="h-4 w-4 text-rose-400 inline" />
                      <span>GATE STATUS: REJECTED (SAFETY LOCK ENGAGED)</span>
                    </>
                  )}
                </div>
                <p className="text-slate-300 font-mono text-[11px]">
                  <strong>Reason:</strong> {validationResult.reason}
                </p>
              </div>
            )}

            {/* Caries Model Execution Gate */}
            <div className="pt-2 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs">
                {validationResult?.accepted ? (
                  <div className="flex items-center gap-1.5 text-emerald-400">
                    <Unlock className="h-4 w-4" />
                    <span className="font-semibold">Caries Model Safety Gate: UNLOCKED</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 text-rose-400">
                    <Lock className="h-4 w-4" />
                    <span className="font-semibold">Caries Model Safety Gate: LOCKED</span>
                  </div>
                )}
              </div>

              <button
                onClick={runCariesInference}
                disabled={!validationResult?.accepted || isClassifying}
                className="w-full sm:w-auto px-4 py-2 rounded-xl text-xs font-bold transition flex items-center justify-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed bg-emerald-600 hover:bg-emerald-500 text-slate-950"
              >
                {isClassifying ? (
                  <>
                    <div className="h-3.5 w-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                    <span>Classifying Surfaces...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3.5 w-3.5" />
                    <span>Run Caries Classification</span>
                  </>
                )}
              </button>
            </div>

            {/* Caries Classification Output */}
            {cariesResult && (
              <div className="p-3.5 rounded-xl bg-slate-950 border border-emerald-500/30 text-xs space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="h-4 w-4" /> Caries Surface Model Output
                  </span>
                  <span className="font-mono text-[10px] text-slate-400">{cariesResult.model}</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] pt-1">
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-400 block">Surfaces Examined</span>
                    <strong className="text-white font-mono">{cariesResult.results?.surfacesExamined}</strong>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-400 block">Overall Risk</span>
                    <strong className="text-amber-400 font-mono">{cariesResult.results?.overallRiskScore}</strong>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-400 block">Mean Confidence</span>
                    <strong className="text-emerald-400 font-mono">
                      {((cariesResult.results?.confidenceMean || 0) * 100).toFixed(1)}%
                    </strong>
                  </div>
                </div>
                <div className="mt-2 space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold">Detected Lesions:</span>
                  {cariesResult.results?.detectedLesions.map((lesion, idx) => (
                    <div key={idx} className="flex items-center justify-between p-1.5 rounded bg-slate-900/60 font-mono text-[10px] text-slate-300">
                      <span><strong>Tooth:</strong> {lesion.tooth} ({lesion.surface})</span>
                      <span className="text-amber-300">{lesion.severity} ({(lesion.confidence * 100).toFixed(0)}%)</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
