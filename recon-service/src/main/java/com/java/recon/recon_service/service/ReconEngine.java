package com.java.recon.recon_service.service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.avro.generic.GenericRecord;
import com.java.recon.recon_service.model.BreakRecord;

public class ReconEngine {
    public List<BreakRecord> reconcile(
        List<Map<String,String>> excel,
        List<GenericRecord> parquet,
        String key) {

        Map<String, GenericRecord> parquetMap =
            parquet.stream()
                .collect(Collectors.toMap(
                    r -> r.get(key).toString(),
                    r -> r));

        List<BreakRecord> breaks = new ArrayList<>();

        for (int i = 0; i < excel.size(); i++) {
            Map<String,String> row = excel.get(i);
            String id = row.get(key);

            GenericRecord p = parquetMap.get(id);

            if (p == null) {
                breaks.add(new BreakRecord(i+1, key, id, "MISSING",
                        "Row not found in parquet"));
                continue;
            }

            for (String col : row.keySet()) {
                String expected = row.get(col);
                String actual = p.get(col).toString();

                if (!expected.equals(actual)) {
                    breaks.add(new BreakRecord(i+1, col,
                            expected, actual, "Mismatch"));
                }
            }
        }
        return breaks;
    }

}