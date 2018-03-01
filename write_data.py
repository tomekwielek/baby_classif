import xlsxwriter

workbook = xlsxwriter.Workbook('write_data.xlsx')
worksheet = workbook.add_worksheet()


row = 0
col = 0

idfs = names #change
for key in idfs.keys():
    row += 1
    worksheet.write(row, col, key)
    for item in idfs[key]:
        worksheet.write(row, col + 1, item)
        row += 1
workbook.close()


for item in names:
    worksheet.write(0, col, item)
    col += 1
workbook.close()
