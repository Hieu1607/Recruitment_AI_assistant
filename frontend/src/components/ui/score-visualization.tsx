import { cn } from "@/lib/cn";
import {
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
} from "recharts";

export interface ScoreSegment {
  label: string;
  value: number;
  color?: string;
}

export interface RadarDataPoint {
  subject: string;
  value: number;
  fullMark?: number;
}

const DEFAULT_SEGMENT_COLORS = [
  "#1F3A2E",
  "#2A5A78",
  "#5A3A7E",
  "#7A3A3A",
  "#3A5A3A",
  "#5A4A2A",
];

const RADAR_LABEL_MIN_GUTTER = 72;

function getScoreColor(score: number): string {
  if (score >= 80) return "var(--success)";
  if (score >= 60) return "var(--warning)";
  return "var(--danger)";
}

function wrapRadarLabel(label: string, maxCharsPerLine: number): string[] {
  const normalized = label.replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  const words = normalized.split(" ");
  const lines: string[] = [];
  let currentLine = "";

  for (const word of words) {
    const candidate = currentLine ? `${currentLine} ${word}` : word;
    if (candidate.length <= maxCharsPerLine) {
      currentLine = candidate;
      continue;
    }

    if (currentLine) {
      lines.push(currentLine);
    }
    currentLine = word;
  }

  if (currentLine) {
    lines.push(currentLine);
  }

  return lines;
}

function createRadarAxisTick(size: number) {
  const labelGutter = Math.max(RADAR_LABEL_MIN_GUTTER, Math.round(size * 0.2));
  const maxCharsPerLine = Math.max(16, Math.round(labelGutter / 6.8));
  const horizontalOffset = Math.max(8, Math.round(size * 0.02));
  const lineHeight = 13;

  return function RadarAxisTick(props: any) {
    const { x = 0, y = 0, textAnchor = "middle", payload } = props as {
      x?: number | string;
      y?: number | string;
      textAnchor?: "inherit" | "start" | "middle" | "end";
      payload?: { value?: string };
    };
    const fullLabel = String(payload?.value ?? "").trim();
    const lines = wrapRadarLabel(fullLabel, maxCharsPerLine);
    const textOffset = textAnchor === "start" ? horizontalOffset : textAnchor === "end" ? -horizontalOffset : 0;
    const firstLineDy = -((lines.length - 1) * lineHeight) / 2;

    return (
      <g transform={`translate(${Number(x)},${Number(y)})`}>
        <text
          x={textOffset}
          y={0}
          textAnchor={textAnchor}
          fontSize={11}
          fontFamily="var(--font-sans)"
          fill="var(--fg-muted)"
        >
          <title>{fullLabel}</title>
          {lines.map((line, index) => (
            <tspan
              key={`${line}-${index}`}
              x={textOffset}
              dy={index === 0 ? firstLineDy : lineHeight}
            >
              {line}
            </tspan>
          ))}
        </text>
      </g>
    );
  };
}

/* ── Mini Bar ─────────────────────────────────────────────────── */

export interface ScoreBarProps {
  score: number;
  size?: "sm" | "md";
  showLabel?: boolean;
  className?: string;
}

export function ScoreBar({ score, size = "md", showLabel = true, className }: ScoreBarProps) {
  const clampedScore = Math.max(0, Math.min(100, score));
  const barHeight = size === "sm" ? "h-0.5" : "h-1";

  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <div
        className={cn("relative rounded-full overflow-hidden bg-[color:var(--hairline)]", barHeight)}
        style={{ width: 80 }}
        role="progressbar"
        aria-valuenow={clampedScore}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={cn("absolute inset-y-0 left-0 rounded-full transition-[width]")}
          style={{
            width: `${clampedScore}%`,
            backgroundColor: getScoreColor(clampedScore),
          }}
        />
      </div>
      {showLabel && (
        <span className="font-mono tabular-nums text-xs text-fg-muted">{clampedScore}%</span>
      )}
    </div>
  );
}

/* ── Donut ────────────────────────────────────────────────────── */

export interface ScoreDonutProps {
  score: number;
  segments?: ScoreSegment[];
  size?: number;
  className?: string;
}

export function ScoreDonut({ score, segments, size = 200, className }: ScoreDonutProps) {
  const clampedScore = Math.max(0, Math.min(100, score));
  const innerRadius = size * 0.3;
  const outerRadius = size * 0.4;

  const pieData = segments && segments.length > 0
    ? segments
    : [
        { label: "Score", value: clampedScore, color: getScoreColor(clampedScore) },
        { label: "Remaining", value: 100 - clampedScore, color: "var(--hairline)" },
      ];

  return (
    <div className={cn("flex flex-col items-center gap-3", className)}>
      <div className="relative" style={{ width: size, height: size }}>
        <PieChart width={size} height={size}>
          <Pie
            data={pieData}
            cx={size / 2}
            cy={size / 2}
            innerRadius={innerRadius}
            outerRadius={outerRadius}
            dataKey="value"
            startAngle={90}
            endAngle={-270}
            strokeWidth={0}
          >
            {pieData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.color ?? DEFAULT_SEGMENT_COLORS[i % DEFAULT_SEGMENT_COLORS.length]}
              />
            ))}
          </Pie>
        </PieChart>
        {/* Center label */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none"
        >
          <span
            className="font-mono tabular-nums font-bold text-fg leading-none"
            style={{ fontSize: size * 0.14 }}
          >
            {clampedScore}
          </span>
          <span
            className="font-sans text-fg-subtle mt-0.5"
            style={{ fontSize: size * 0.065 }}
          >
            / 100
          </span>
        </div>
      </div>
      {/* Legend */}
      {segments && segments.length > 0 && (
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1">
          {segments.map((seg, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <span
                className="inline-block h-2 w-2 rounded-full shrink-0"
                style={{ backgroundColor: seg.color ?? DEFAULT_SEGMENT_COLORS[i % DEFAULT_SEGMENT_COLORS.length] }}
              />
              <span className="font-sans text-[11px] text-fg-muted">{seg.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Radar ────────────────────────────────────────────────────── */

export interface ScoreRadarProps {
  data: RadarDataPoint[];
  size?: number;
  className?: string;
}

export function ScoreRadar({ data, size = 400, className }: ScoreRadarProps) {
  const tick = createRadarAxisTick(size);
  const labelGutter = Math.max(RADAR_LABEL_MIN_GUTTER, Math.round(size * 0.2));
  const outerRadius = Math.max(72, Math.round(size * 0.22));

  return (
    <div className={cn("mx-auto w-full", className)} style={{ maxWidth: size }}>
      <ResponsiveContainer width="100%" height={size}>
        <RadarChart
          data={data}
          cx="50%"
          cy="50%"
          margin={{
            top: labelGutter,
            right: labelGutter,
            bottom: labelGutter,
            left: labelGutter,
          }}
          outerRadius={outerRadius}
        >
          <PolarGrid stroke="var(--hairline)" />
          <PolarAngleAxis
            dataKey="subject"
            tick={tick}
          />
          <Radar
            name="Score"
            dataKey="value"
            stroke="var(--accent)"
            strokeWidth={2}
            fill="var(--accent)"
            fillOpacity={0.18}
          />
          <RechartsTooltip
            contentStyle={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--hairline-strong)",
              borderRadius: "var(--radius-sm)",
              fontSize: 12,
              fontFamily: "var(--font-sans)",
              color: "var(--fg)",
            }}
            cursor={false}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
