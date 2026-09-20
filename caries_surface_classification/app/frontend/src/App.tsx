import React, { useState, useRef } from 'react';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictionResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      // Backend schema: { status: "success"|"error", findings: [...], errors: [...], ... }
      if (!response.ok) {
        // HTTP-level error (4xx/5xx)
        setError(data.error || data.reason || data.detail || `Server error ${response.status}`);
        if (data.validation) {
          setPredictionResult({ validation: data.validation, failed: true });
        }
      } else if (data.status === 'error') {
        // Pipeline-level hard error with no findings
        const errMsg = data.errors?.map((e: any) => e.message).join(' | ')
          || 'An unknown error occurred during prediction.';
        setError(errMsg);
        if (data.validation) {
          setPredictionResult({ validation: data.validation, failed: true });
        }
      } else {
        // status === "success" — pipeline ran (findings may be empty, soft errors are warnings only)
        setPredictionResult({ ...data, success: true });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-800 tracking-tight">Clinical AI: Caries Detection</h1>
            <p className="text-slate-500 mt-1">Panoramic Dental Radiograph Analysis System</p>
          </div>
          <div className="px-4 py-2 bg-emerald-50 text-emerald-700 rounded-full text-sm font-semibold border border-emerald-200">
            System Online
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Column: Upload & Original */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
              <h2 className="text-lg font-semibold mb-4">Input Radiograph</h2>
              
              <div 
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${previewUrl ? 'border-indigo-300 bg-indigo-50/30' : 'border-slate-300 hover:bg-slate-50'}`}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  className="hidden" 
                  ref={fileInputRef} 
                  onChange={handleImageChange}
                  accept="image/png, image/jpeg"
                />
                
                {previewUrl ? (
                  <div className="space-y-4">
                    <img src={previewUrl} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-sm" />
                    <p className="text-sm font-medium text-indigo-600">Click to change image</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="w-16 h-16 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center mx-auto">
                      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-base font-medium text-slate-700">Upload Panoramic X-Ray</p>
                      <p className="text-sm text-slate-500 mt-1">PNG or JPEG</p>
                    </div>
                  </div>
                )}
              </div>
              
              <button
                onClick={handlePredict}
                disabled={!selectedImage || isLoading}
                className={`mt-6 w-full py-3 px-4 rounded-xl font-semibold text-white transition-all shadow-sm ${
                  !selectedImage || isLoading 
                    ? 'bg-slate-300 cursor-not-allowed' 
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-md active:scale-[0.98]'
                }`}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Processing Analysis...</span>
                  </div>
                ) : 'Run Clinical Analysis'}
              </button>
            </div>

            {error && (
              <div className="bg-red-50 text-red-700 p-4 rounded-xl border border-red-200 shadow-sm flex items-start space-x-3">
                <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <h3 className="font-semibold">Analysis Failed</h3>
                  <p className="text-sm mt-1">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="space-y-6">
            
            {/* Validation Gate Section */}
            {predictionResult?.validation && (
              <div className={`p-5 rounded-2xl border shadow-sm ${predictionResult.failed ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
                <div className="flex justify-between items-center mb-3">
                  <h3 className={`font-semibold ${predictionResult.failed ? 'text-red-800' : 'text-emerald-800'}`}>
                    Medical Input Validation
                  </h3>
                  <span className={`px-2 py-1 rounded text-xs font-bold ${predictionResult.failed ? 'bg-red-200 text-red-800' : 'bg-emerald-200 text-emerald-800'}`}>
                    {predictionResult.failed ? 'REJECTED' : 'PASSED'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-white/60 p-3 rounded-lg">
                    <div className="text-slate-500 mb-1">Detection Confidence</div>
                    <div className="font-mono font-medium">{(predictionResult.validation.confidence * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-white/60 p-3 rounded-lg">
                    <div className="text-slate-500 mb-1">ROI Coverage</div>
                    <div className="font-mono font-medium">{(predictionResult.validation.coverageRatio * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            )}

            {/* Inference Results */}
            {predictionResult?.success && (
              <>
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                  <h2 className="text-lg font-semibold mb-4">Annotated Radiograph</h2>
                  <div className="bg-slate-100 rounded-xl overflow-hidden border border-slate-200">
                    <img 
                      src={predictionResult.processedImage} 
                      alt="Annotated Results" 
                      className="w-full h-auto"
                    />
                  </div>
                </div>

                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                  <div className="flex justify-between items-end mb-4">
                    <h2 className="text-lg font-semibold">Detected Findings</h2>
                    <span className="text-sm text-slate-500 font-medium">
                      Total: {predictionResult.findings.length}
                    </span>
                  </div>
                  
                  {predictionResult.findings.length > 0 ? (
                    <div className="overflow-x-auto rounded-xl border border-slate-200">
                      <table className="w-full text-sm text-left">
                        <thead className="bg-slate-50 text-slate-600 font-medium border-b border-slate-200">
                          <tr>
                            <th className="px-4 py-3">FDI Tooth</th>
                            <th className="px-4 py-3">Surface</th>
                            <th className="px-4 py-3">RF Confidence</th>
                            <th className="px-4 py-3">YOLO Confidence</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {predictionResult.findings.map((pred: any, i: number) => (
                            <tr key={pred.detection_id || i} className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-mono">{pred.fdi || 'N/A'}</td>
                              <td className="px-4 py-3 text-slate-700 capitalize">{pred.surface.replace('_', ' ')}</td>
                              <td className="px-4 py-3">
                                <div className="flex items-center space-x-2">
                                  <div className="w-16 h-2 bg-slate-200 rounded-full overflow-hidden">
                                    <div 
                                      className="h-full bg-indigo-500"
                                      style={{ width: `${(pred.rf_confidence || 0) * 100}%` }}
                                    />
                                  </div>
                                  <span className="font-mono text-xs text-slate-500">
                                    {((pred.rf_confidence || 0) * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </td>
                              <td className="px-4 py-3">
                                <div className="flex items-center space-x-2">
                                  <div className="w-16 h-2 bg-slate-200 rounded-full overflow-hidden">
                                    <div 
                                      className="h-full bg-emerald-500" 
                                      style={{ width: `${pred.yolo_confidence * 100}%` }}
                                    />
                                  </div>
                                  <span className="font-mono text-xs text-slate-500">
                                    {(pred.yolo_confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center py-8 bg-slate-50 rounded-xl border border-slate-200 text-slate-500">
                      No caries detected in this radiograph.
                    </div>
                  )}
                </div>

                {/* Resource Usage */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800 text-slate-300 p-4 rounded-xl flex items-center justify-between">
                    <div className="text-sm">Inference Latency</div>
                    <div className="font-mono text-white font-medium">{predictionResult.summary?.inference_latency_ms || 0}ms</div>
                  </div>
                  <div className="bg-slate-800 text-slate-300 p-4 rounded-xl flex items-center justify-between">
                    <div className="text-sm">Detections Evaluated</div>
                    <div className="font-mono text-white font-medium">{predictionResult.summary?.valid_detection_count || 0}</div>
                  </div>
                </div>
              </>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
