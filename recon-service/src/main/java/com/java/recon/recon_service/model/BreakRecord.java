package com.java.recon.recon_service.model;

import lombok.Data;
import lombok.AllArgsConstructor;

@Data
@AllArgsConstructor
public class BreakRecord {
    int lineNo;
    String attribute;
    String expected;
    String actual;
    String reason;
}