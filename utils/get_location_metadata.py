import wbgapi as wb

for row in wb.data.fetch('SP.POP.TOTL', 'USA'):
    print(row)