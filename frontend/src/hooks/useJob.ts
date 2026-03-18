import { useState, useCallback, useRef } from "react";
import type { JobState, TileStatusEvent, TileStatus } from "../types";
import { useSSE } from "./useSSE";
import { getProgressUrl } from "../api/client";

const INITIAL_STATE: JobState = {
  jobId: null,
  tileGrid: [],
  currentTile: null,
  completed: 0,
  total: 0,
  elapsed: 0,
  finished: false,
  tileTimes: [],
};

export function useJob() {
  const [state, setState] = useState<JobState>(INITIAL_STATE);
  const lastTileTime = useRef<number>(0);

  const sseUrl =
    state.jobId && !state.finished
      ? getProgressUrl(state.jobId)
      : null;

  const startJob = useCallback(
    (jobId: string, gridI: number, gridJ: number) => {
      const grid: TileStatus[][] = Array.from({ length: gridI }, () =>
        Array.from({ length: gridJ }, () => "pending" as TileStatus)
      );
      lastTileTime.current = Date.now();
      setState({
        jobId,
        tileGrid: grid,
        currentTile: null,
        completed: 0,
        total: 0,
        elapsed: 0,
        finished: false,
        tileTimes: [],
      });
    },
    []
  );

  const onTile = useCallback((data: unknown) => {
    const event = data as TileStatusEvent;
    setState((prev) => {
      const grid = prev.tileGrid.map((row) => [...row]);

      if (event.status === "in_flight") {
        if (grid[event.i] && grid[event.i][event.j] !== "completed") {
          grid[event.i][event.j] = "in_flight";
        }
        return {
          ...prev,
          tileGrid: grid,
          currentTile: { i: event.i, j: event.j, k: event.k },
          elapsed: event.elapsed,
          total: event.total,
        };
      }

      // completed or failed
      if (event.status === "completed" || event.status === "failed") {
        if (grid[event.i]) {
          // Mark completed only when all k tiles for this (i,j) are done
          // For simplicity, use the latest status
          grid[event.i][event.j] = event.status;
        }

        const now = Date.now();
        const tileTime = (now - lastTileTime.current) / 1000;
        lastTileTime.current = now;

        return {
          ...prev,
          tileGrid: grid,
          completed: event.completed,
          total: event.total,
          elapsed: event.elapsed,
          tileTimes:
            event.status === "completed"
              ? [...prev.tileTimes, tileTime]
              : prev.tileTimes,
        };
      }

      return prev;
    });
  }, []);

  const onComplete = useCallback(() => {
    setState((prev) => ({ ...prev, finished: true }));
  }, []);

  const onDone = useCallback(() => {
    setState((prev) => ({ ...prev, finished: true, currentTile: null }));
  }, []);

  useSSE({ url: sseUrl, onTile, onComplete, onDone });

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  return { state, startJob, reset };
}
