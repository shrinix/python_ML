# Excel 3-Icon Conditional Formatting Tool

Apply a 3-icon conditional format to a range in an existing `.xlsx` workbook. The rule splits values into three buckets around zero:

- top icon: > 0
- middle icon: = 0
- bottom icon: < 0

Supported icon styles include: 3Arrows, 3ArrowsGray, 3Flags, 3TrafficLights1, 3TrafficLights2, 3Signs, 3Symbols, 3Symbols2, 3Stars, 3Triangles.

## Install

```bash
python -m pip install --upgrade openpyxl
```

## Usage (Icon Set)

```bash
python python_utils/excel_utils/excel_conditional_format.py \
  --mode iconset \
  --file /path/to/workbook.xlsx \
  --sheet "Sheet1" \
  --range "B2:F100" \
  --icon-style 3Symbols2 \
  --icons-only                 # hides numbers and shows icons only
  --coerce-numbers             # convert values in range to numbers before formatting
  --coerce-non-numeric-zero    # set unparsable cells to 0
```

- Use `--reverse` to flip icon order.
- `--icons-only` or `--hide-values` hides numeric values under icons.

## Usage (Number Format Icons)

To exactly map: orange flag for >0, green check for =0, red X for <0,
use the custom number-format mode (this does not use Excel's Icon Sets):

```bash
python python_utils/excel_utils/excel_conditional_format.py \
  --mode number-format \
  --file /path/to/workbook.xlsx \
  --sheet "Sheet1" \
  --range "B2:F100" \
  --nf-pattern "0"   # or "0.00", "#,##0"
  --nf-icons-only     # replace values entirely with icons
  --nf-pos-symbol "▲" \
  --nf-zero-symbol "⚑" \
  --nf-neg-symbol "✗" \
  --coerce-numbers    # convert values to numbers first
  --coerce-non-numeric-zero
```

- Positive (>0): orange flag (approx) + value
- Zero (=0): green check + value
- Negative (<0): red X + value

Note: Colors use Excel's built-in [Color46] for orange.

## Custom IconSet (Excel-only)

If you need a true Custom IconSet (e.g., green check + orange flag + red cross) that appears in Excel's Conditional Formatting Manager:

- openpyxl cannot write Custom IconSets; it supports only built-in sets.
- Use the provided xlwings-based tool to automate Excel:

```bash
python -m pip install --upgrade -r /Users/shriniwasiyengar/git/python_utils/requirements.txt

python /Users/shriniwasiyengar/git/python_utils/excel_utils/excel_custom_iconset_xlwings.py \
  --file /path/to/workbook.xlsx \
  --sheet "Sheet1" \
  --range "B2:B11"
```

This sets thresholds around zero and assigns icons:
- >0: green check (up)
- =0: orange/yellow flag
- <0: red cross

Requires Microsoft Excel (macOS or Windows). On macOS, grant Excel automation permissions if prompted.

## Notes

- The script modifies the workbook in place. Consider working on a copy if needed.
- Icon set buckets are implemented via thresholds `[0, 0]` (numeric), a common pattern to split negatives, zero, and positives.
- Excel Icon Sets cannot mix different icon families (e.g., flags + checks + X) in one rule when using openpyxl. The number-format mode is provided to achieve that exact combination without relying on Excel. Alternatively, use the xlwings script above to create a true Custom IconSet via Excel.
- Coercion parses common formats like `$1,234`, `(5)`, `42%`, and converts to numbers; non-numeric cells remain unchanged unless `--coerce-non-numeric-zero` is set.
 - Stronger default colors are used: `[Color4]` (Bright Green) for positive, `[Color46]` (Orange) for zero, `[Color3]` (Red) for negative. You can still override via `--nf-*-color`.
 - Customize icons and colors via `--nf-pos-symbol`, `--nf-zero-symbol`, `--nf-neg-symbol` and `--nf-*-color` flags to match your desired visual mapping.
