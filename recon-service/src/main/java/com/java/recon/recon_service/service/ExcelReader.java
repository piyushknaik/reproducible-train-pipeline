package com.java.recon.recon_service.service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.Sheet;

public class ExcelReader {
 

    public List<Map<String, String>> readExcel(String path) throws Exception {
        Workbook wb = new XSSFWorkbook(new FileInputStream(path));
        Sheet sheet = wb.getSheetAt(0);

        List<Map<String, String>> rows = new ArrayList<>();

        Row header = sheet.getRow(0);

        for (int i = 1; i <= sheet.getLastRowNum(); i++) {
            Row row = sheet.getRow(i);
            Map<String, String> map = new HashMap<>();

            for (int j = 0; j < header.getLastCellNum(); j++) {
                map.put(header.getCell(j).getStringCellValue(),
                        row.getCell(j).toString());
            }
            rows.add(map);
        }
        return rows;
    }
}