python /Users/shriniwasiyengar/git/python_utils/excel_utils/create_sample_excel.py

python /Users/shriniwasiyengar/git/python_utils/excel_utils/excel_conditional_format.py \
  --mode number-format \
  --file /Users/shriniwasiyengar/git/python_utils/excel_utils/sample.xlsx \
  --sheet 'Sheet1' \
  --range 'B2:B11' \
  --nf-icons-only \
  --nf-add-placeholder-iconset \
  --icon-style '3Symbols2' \
  --nf-pos-symbol '✓↑' \
  --nf-zero-symbol '⚑' \
  --nf-neg-symbol '✗' \
  --nf-pos-color 'Color4' \
  --nf-zero-color 'Color46' \
  --nf-neg-color 'Color3' \
  --coerce-numbers \
  --coerce-non-numeric-zero