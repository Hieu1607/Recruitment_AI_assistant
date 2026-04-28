# Phase 2: Primitives Library — UI Design Contract

**Gathered:** 2026-04-28
**Status:** Approved
**Phase:** 02-primitives-library

---

## 1. Typography

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Button label (md) | `font-sans` (Geist) | 14px / 0.875rem | 500 | `--fg` or `--accent-fg` |
| Button label (sm) | `font-sans` | 12px / 0.75rem | 500 | same |
| Button label (lg) | `font-sans` | 15px / 0.9375rem | 500 | same |
| Badge label | `font-sans` | 11px / 0.6875rem | 500 | variant-specific |
| Avatar initials | `font-sans` | 12px (sm) / 14px (md) / 18px (lg) | 600 | `--accent-fg` |
| Filter chip | `font-sans` | 13px / 0.8125rem | 400→500 active | `--fg-muted`→`--fg` |
| Table header | `font-sans` | 12px / 0.75rem | 500 uppercase | `--fg-subtle` |
| Table cell | `font-sans` | 13px / 0.8125rem | 400 | `--fg` |
| EmptyState heading | `font-display` (Fraunces) | 20px / 1.25rem | 500 | `--fg` |
| EmptyState body | `font-sans` | 14px / 0.875rem | 400 | `--fg-muted` |
| Pagination | `font-sans` | 13px / 0.8125rem | 400 | `--fg-muted` |
| Tooltip | `font-sans` | 12px / 0.75rem | 400 | `--fg` |
| Score labels | `font-mono` (Geist Mono) | 11–13px | 400 | `--fg-muted` |
| Score numerals | `font-mono` tabular-nums | 13–24px | 600 | `--fg` |

---

## 2. Color Palette (from tokens.css)

### Status Badge Colors

| Status | Background | Text | Dot |
|--------|-----------|------|-----|
| `pending` / `queued` | `rgba(138,138,133,0.12)` | `--fg-muted` | `--neutral` |
| `processing` / `running` | `rgba(184,138,62,0.14)` | `#B88A3E` | `--warning` |
| `completed` / `active` / `passed` | `rgba(74,124,89,0.14)` | `--success` | `--success` |
| `failed` / `danger` | `rgba(184,68,46,0.14)` | `--danger` | `--danger` |
| `sent` | `rgba(74,124,89,0.14)` | `--success` | `--success` |
| `not_sent` / `draft` | `rgba(138,138,133,0.12)` | `--fg-muted` | `--neutral` |

### Button Colors

| Variant | Background | Text | Hover | Border |
|---------|-----------|------|-------|--------|
| `primary` | `--accent` `#1F3A2E` | `--accent-fg` | `--accent-hover` | none |
| `secondary` | `--bg-elevated` | `--fg` | `--bg-sidebar` | `1px --hairline-strong` |
| `ghost` | transparent | `--fg-muted` | `--hairline` bg | none |
| `danger` | `rgba(184,68,46,0.1)` | `--danger` | `rgba(184,68,46,0.18)` | `1px rgba(184,68,46,0.3)` |
| `icon` (ghost) | transparent | `--fg-muted` | `--hairline` bg | none |

---

## 3. Spacing & Sizing

### Button Sizes (8px grid)

| Size | Height | Padding (h×v) | Icon size | Border-radius |
|------|--------|---------------|-----------|---------------|
| `sm` | 28px | 8px × 12px | 14px | `--radius-sm` 4px |
| `md` | 36px | 10px × 16px | 16px | `--radius-md` 6px |
| `lg` | 44px | 12px × 20px | 18px | `--radius-md` 6px |

### Avatar Sizes

| Size | Diameter | Font-size |
|------|----------|-----------|
| `sm` | 24px | 10px |
| `md` | 32px | 13px |
| `lg` | 40px | 16px |
| `xl` | 56px | 22px |

### Filter Chip
- Height: 28px
- Padding: 6px × 12px
- Border-radius: `--radius-full` (pill)
- Border: `1px solid --hairline` inactive → `1px solid --accent` active

### DataTable
- Row height: 48px
- Header height: 40px
- Checkbox col width: 40px
- Min column width: 80px

### Modal
- Max width: 560px (default), 720px (large)
- Border-radius: `--radius-lg` 10px
- Padding: 24px
- Backdrop: `rgba(0,0,0,0.4)` blur-sm

### Tooltip
- Max width: 240px
- Padding: 6px × 10px
- Border-radius: `--radius-sm` 4px
- Border: `1px solid --hairline-strong`
- Shadow: `--shadow-md`
- Delay: 400ms open, 0ms close

### Skeleton
- Background: `--hairline` → `--hairline-strong`
- Animation: shimmer 1.5s ease-in-out infinite (linear-gradient sweep)
- Row height: 16px (text), 48px (table row), 200px (card)
- Border-radius: `--radius-sm`

### Pagination
- Height: 36px
- Prev/Next buttons: 32px square
- Gap between items: 4px

---

## 4. Motion

| Component | Property | Duration | Easing |
|-----------|---------|---------|--------|
| Button press | scale | 100ms | ease-out |
| Modal open/close | opacity + translateY(8px) | 200ms | `--ease-out` |
| Toast enter | translateX + opacity | 280ms | `--ease-out` |
| Toast exit | opacity + scale | 200ms | ease-in |
| Tooltip open | opacity | 120ms | ease-out |
| Dropdown/menu | opacity + scale(0.97) | 140ms | `--ease-out` |
| Skeleton shimmer | background-position | 1500ms | ease-in-out infinite |
| FilterChip toggle | background-color + border | 120ms | ease-out |

---

## 5. Focus & Accessibility

- **Focus ring**: `2px solid --accent`, `2px offset`, for all interactive elements
- All interactive components expose proper ARIA roles (via Radix UI primitives)
- Keyboard navigation on DataTable (Tab through cells, Space for checkbox)
- Tooltip accessible via `aria-describedby`
- Modal uses `role="dialog"` with `aria-modal="true"` and focus trap
- FilterChip: `role="checkbox"` or `role="option"` depending on selection mode

---

## 6. ScoreVisualization Specs

### Mini Bar (inline)
- Height: 4px bar, width: 80px container
- Accent fill color on neutral track
- No labels at this size

### Donut (200px)
- SVG 200×200, stroke-width 16
- Center: numeric score in `font-mono` tabular-nums, 28px bold
- Legend: 3-5 segments labeled below in 11px sans

### Radar (400px)
- SVG 400×400
- 4-6 axes matching scoring criteria
- Fill: `rgba(31,58,46,0.2)` (accent at 20%), stroke `--accent`
- Axis labels: 11px `font-sans` `--fg-muted`

---

## UI-SPEC VERIFIED

All 6 dimensions assessed:
1. ✓ Typography — complete token mapping per component
2. ✓ Color — all variants + status mapping + dark-mode parity via CSS tokens
3. ✓ Spacing — 8px grid, all sizes documented
4. ✓ Motion — durations + easing per component
5. ✓ Accessibility — focus rings, ARIA, keyboard nav
6. ✓ ScoreVisualization — three sizes fully spec'd
