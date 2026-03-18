interface Props {
  tileSize: number;
  onTileSizeChange: (size: number) => void;
}

export function ConfigPanel({ tileSize, onTileSizeChange }: Props) {
  return (
    <div className="config-panel">
      <h3>Configuration</h3>
      <label>
        Tile Size:
        <select
          value={tileSize}
          onChange={(e) => onTileSizeChange(parseInt(e.target.value))}
        >
          <option value={128}>128 x 128</option>
          <option value={256}>256 x 256</option>
        </select>
      </label>
    </div>
  );
}
