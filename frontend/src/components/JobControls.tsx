interface Props {
  canStart: boolean;
  isRunning: boolean;
  onStart: () => void;
  onCancel: () => void;
}

export function JobControls({ canStart, isRunning, onStart, onCancel }: Props) {
  return (
    <div className="job-controls">
      {!isRunning ? (
        <button
          className="btn-start"
          onClick={onStart}
          disabled={!canStart}
        >
          Start Multiplication
        </button>
      ) : (
        <button className="btn-cancel" onClick={onCancel}>
          Cancel
        </button>
      )}
    </div>
  );
}
