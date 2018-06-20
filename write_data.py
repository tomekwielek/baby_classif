def mywriter(data, save_name):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(save_name)
    worksheet = workbook.add_worksheet()
    save_name = save_name + '.xlsx'
    row = 0
    col = 0

    for key in data.keys():
        row += 1
        worksheet.write(row, col, key)
        for item in data[key]:
            worksheet.write(row, col + 1, item)
            #row += 1
    workbook.close()


    for item in data:
        worksheet.write(0, col, item)
        col += 1
    workbook.close()
    return
