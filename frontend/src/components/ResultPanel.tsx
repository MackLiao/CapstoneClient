import { useState } from "react";
import type { VerificationResult } from "../types";
import { verifyJob } from "../api/client";

interface Props {
  jobId: string | null;
  finished: boolean;
}

export function ResultPanel({ jobId, finished }: Props) {
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!jobId || !finished) return null;

  const handleVerify = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await verifyJob(jobId);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Verification failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="result-panel">
      <h3>Results</h3>
      {!result ? (
        loading ? (
          <div className="skeleton-bar" />
        ) : (
          <button onClick={handleVerify}>Verify Against NumPy</button>
        )
      ) : (
        <div className={`verification ${result.passed ? "passed" : "failed"}`}>
          <p className="verdict">
            {result.passed ? "PASSED" : "FAILED"}
          </p>
          <table>
            <tbody>
              <tr>
                <td>Max Absolute Error:</td>
                <td>{result.max_abs_error.toExponential(4)}</td>
              </tr>
              <tr>
                <td>Mean Absolute Error:</td>
                <td>{result.mean_abs_error.toExponential(4)}</td>
              </tr>
              <tr>
                <td>Tolerance:</td>
                <td>{result.tolerance}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
