export interface HeartbeatResponse {
  status: 'alive' | 'error' | string;
  timestamp: string;
}

export interface FormattedBytes {
  raw: number;
  mb: number;
  formatted: string;
}

export interface SystemMetricsResponse {
  timestamp: string;
  uptime: {
    seconds: number;
    formatted: string;
  };
  memory: {
    rss: FormattedBytes;
    heapTotal: FormattedBytes;
    heapUsed: FormattedBytes;
    external: FormattedBytes;
    arrayBuffers: FormattedBytes;
    heapUsagePercentage: number;
  };
  cpu: {
    userMicroseconds: number;
    systemMicroseconds: number;
    userTimeMs: number;
    systemTimeMs: number;
    cpuUsagePercentage: number;
  };
  traffic: {
    totalRequests: number;
    activeRequests: number;
    lastLatencyMs: number;
    averageLatencyMs: number;
  };
  nodeVersion: string;
  platform: string;
  pid: number;
}

export interface LogEntry {
  id: number;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';
  method: string;
  path: string;
  status: number;
  durationMs: number;
  message: string;
}

export interface ValidationResult {
  accepted: boolean;
  imageType: string;
  confidence: number;
  coverageRatio: number;
  maskAreaRatio: number;
  validMask: boolean;
  step?: number;
  reason?: string;
}

export interface CariesClassificationResult {
  success: boolean;
  status: string;
  model?: string;
  timestamp?: string;
  error?: string;
  results?: {
    surfacesExamined: number;
    detectedLesions: Array<{
      tooth: string;
      surface: string;
      severity: string;
      confidence: number;
    }>;
    overallRiskScore: string;
    confidenceMean: number;
  };
  validationGateTelemetry?: {
    imageType: string;
    confidence: number;
    coverageRatio: number;
  };
}
