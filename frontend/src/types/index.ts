export interface MatrixInfo {
  id: string;
  rows: number;
  cols: number;
}

export interface JobInfo {
  job_id: string;
}

export type TileStatus = "pending" | "in_flight" | "completed" | "failed";

export interface TileStatusEvent {
  i: number;
  j: number;
  k: number;
  status: TileStatus;
  completed: number;
  total: number;
  elapsed: number;
}

export interface JobCompleteEvent {
  status: string;
  completed: number;
  total: number;
  elapsed: number;
}

export interface VerificationResult {
  max_abs_error: number;
  mean_abs_error: number;
  passed: boolean;
  tolerance: number;
}

export interface JobState {
  jobId: string | null;
  tileGrid: TileStatus[][];  // [I][J] — shows aggregate status for (i,j) block
  currentTile: { i: number; j: number; k: number } | null;
  completed: number;
  total: number;
  elapsed: number;
  finished: boolean;
  tileTimes: number[];  // for ETA calculation
}
