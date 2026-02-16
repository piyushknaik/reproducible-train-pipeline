package com.java.recon.recon_service.service;

import java.io.File;

import org.mlflow.tracking.MlflowClient;

public class MlflowLogger {
    public void logRun(int total, int breaks, File reportFile) {

        MlflowClient client = new MlflowClient();

        RunInfo run = client.createRun("0");

        client.logMetric(run.getRunUuid(), "total_rows", total);
        client.logMetric(run.getRunUuid(), "break_count", breaks);
        client.logMetric(run.getRunUuid(), "break_pct",
                (double) breaks / total);

        client.logArtifact(run.getRunUuid(), reportFile.getPath());

        client.setTerminated(run.getRunUuid());
    }
}