package com.java.recon.recon_service.service;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.apache.avro.generic.GenericRecord;

public class ParquetReader {
    public List<GenericRecord> readParquet(String path) throws IOException {
        Path file = new Path(path);

        ParquetReader<GenericRecord> reader =
            AvroParquetReader.<GenericRecord>builder(file).build();

        List<GenericRecord> records = new ArrayList<>();

        GenericRecord record;
        while ((record = reader.read()) != null) {
            records.add(record);
        }
        return records;
    }
}