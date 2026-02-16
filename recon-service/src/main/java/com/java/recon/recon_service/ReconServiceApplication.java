package com.java.recon.recon_service;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ReconServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(ReconServiceApplication.class, args);

		var excel = excelReader.readExcel("stock_reference.xlsx");
		var parquet = parquetReader.readParquet("stock_system.parquet");

		var breaks = engine.reconcile(excel, parquet, "id");

		File report = writeCsv(breaks);

		mlflowLogger.logRun(excel.size(), breaks.size(), report);

		System.out.println("Breaks found: " + breaks.size());
	}

}
