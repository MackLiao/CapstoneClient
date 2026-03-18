import { useEffect, useRef, useCallback } from "react";

interface UseSSEOptions {
  url: string | null;
  onTile: (data: unknown) => void;
  onComplete: (data: unknown) => void;
  onDone: () => void;
}

export function useSSE({ url, onTile, onComplete, onDone }: UseSSEOptions) {
  const sourceRef = useRef<EventSource | null>(null);

  const close = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!url) return;

    close();

    const es = new EventSource(url);
    sourceRef.current = es;

    es.addEventListener("tile", (e) => {
      try {
        onTile(JSON.parse(e.data));
      } catch {
        // ignore parse errors
      }
    });

    es.addEventListener("complete", (e) => {
      try {
        onComplete(JSON.parse(e.data));
      } catch {
        // ignore
      }
    });

    es.addEventListener("done", () => {
      onDone();
      close();
    });

    es.onerror = () => {
      // EventSource auto-reconnects; if CLOSED, give up
      if (es.readyState === EventSource.CLOSED) {
        close();
      }
    };

    return close;
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps

  return { close };
}
