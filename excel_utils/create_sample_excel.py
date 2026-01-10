from openpyxl import Workbook
from pathlib import Path
p = Path('/Users/shriniwasiyengar/git/python_utils/excel_utils/sample.xlsx')
wb = Workbook()
ws = wb.active
ws.title = 'Sheet1'
# Fill sample numbers including negatives and zeros
values = [ -3, -1, 0, 0, 1, 2, 10, -5, 0, 7 ]
for i, v in enumerate(values, start=2):
    ws[f'B{i}'] = v
wb.save(p)
print('Created', p)