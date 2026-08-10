package com.multimodal.mcp.mcp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.BiFunction;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import com.multimodal.mcp.bird.BirdQueryService;
import com.multimodal.mcp.core.ResultConfig;
import com.multimodal.mcp.core.RuntimeState;
import com.multimodal.mcp.format.FormatMab;
import com.multimodal.mcp.format.FormatName;
import com.multimodal.mcp.format.FormatSelector;
import com.multimodal.mcp.format.OptimizationTarget;
import com.multimodal.mcp.format.SelectionContext;
import com.multimodal.mcp.security.AuthService;
import com.multimodal.mcp.security.FileValidationService;
import com.multimodal.mcp.security.PrivacyService;
import com.multimodal.mcp.tabular.PayloadCache;
import com.multimodal.mcp.tabular.TabularData;
import com.multimodal.mcp.tabular.TabularService;

import io.modelcontextprotocol.server.McpServerFeatures.SyncToolSpecification;
import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import io.modelcontextprotocol.spec.McpSchema.JsonSchema;
import io.modelcontextprotocol.spec.McpSchema.TextContent;
import io.modelcontextprotocol.spec.McpSchema.Tool;

/**
 * Registers all MCP tools for the LargeOutputBenchmark server.
 */
@Component
public class McpToolRegistry {

    private static final Set<String> VALID_ARROW_IPC_COMPRESSIONS = Set.of("none", "lz4", "zstd");

    private final RuntimeState runtimeState;
    private final TabularService tabularService;
    private final PayloadCache payloadCache;
    private final FormatSelector formatSelector;
    private final FormatMab formatMab;
    private final BirdQueryService birdQueryService;
    private final FileValidationService fileValidationService;
    private final AuthService authService;
    private final PrivacyService privacyService;
    private final String serverBaseUrl;

    public McpToolRegistry(
            RuntimeState runtimeState,
            TabularService tabularService,
            PayloadCache payloadCache,
            FormatSelector formatSelector,
            FormatMab formatMab,
            BirdQueryService birdQueryService,
            FileValidationService fileValidationService,
            AuthService authService,
            PrivacyService privacyService,
            @Value("${server.base-url:http://localhost:8000}") String serverBaseUrl) {
        this.runtimeState = runtimeState;
        this.tabularService = tabularService;
        this.payloadCache = payloadCache;
        this.formatSelector = formatSelector;
        this.formatMab = formatMab;
        this.birdQueryService = birdQueryService;
        this.fileValidationService = fileValidationService;
        this.authService = authService;
        this.privacyService = privacyService;
        this.serverBaseUrl = stripTrailingSlash(serverBaseUrl);
    }

    public List<SyncToolSpecification> buildToolSpecifications() {
        List<SyncToolSpecification> specs = new ArrayList<>();
        specs.add(tool("large_json", "Return the full dataset as JSON (baseline).", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "result_id", stringProp("Optional materialized result id")),
                List.of("n_rows", "n_cols")),
                args -> handleLargeJson(args)));

        specs.add(tool("large_parquet_blob", "Prepare a Parquet blob descriptor.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "compression", stringProp("Parquet compression codec"),
                        "encoding_strategy", stringProp("Parquet encoding strategy"),
                        "result_id", stringProp("Optional materialized result id")),
                List.of("n_rows", "n_cols")),
                args -> handleLargeParquetBlob(args)));

        specs.add(tool("large_parquet_stream", "Prepare a Parquet streaming descriptor.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "rows_per_chunk", integerProp("Rows per stream chunk"),
                        "compression", stringProp("Parquet compression codec"),
                        "encoding_strategy", stringProp("Parquet encoding strategy"),
                        "result_id", stringProp("Optional materialized result id")),
                List.of("n_rows", "n_cols", "rows_per_chunk")),
                args -> handleLargeParquetStream(args)));

        specs.add(tool("large_arrow_ipc_blob", "Prepare an Arrow IPC blob descriptor.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "ipc_compression", stringProp("Arrow IPC compression"),
                        "result_id", stringProp("Optional materialized result id")),
                List.of("n_rows", "n_cols")),
                args -> handleLargeArrowIpcBlob(args)));

        specs.add(tool("large_arrow_ipc_stream", "Prepare an Arrow IPC stream descriptor.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "rows_per_chunk", integerProp("Rows per stream chunk"),
                        "ipc_compression", stringProp("Arrow IPC compression"),
                        "result_id", stringProp("Optional materialized result id")),
                List.of("n_rows", "n_cols", "rows_per_chunk")),
                args -> handleLargeArrowIpcStream(args)));

        specs.add(tool("describe_result_formats", "Return format hints for client-driven selection.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "rows_per_chunk", integerProp("Rows per chunk for stream hints"),
                        "result_id", stringProp("Optional materialized result id"),
                        "optimization_target", stringProp("min_bytes | min_latency | min_time_to_first_rows"),
                        "prefer_streaming", booleanProp("Prefer streaming formats")),
                List.of("n_rows", "n_cols")),
                args -> handleDescribeResultFormats(args)));

        specs.add(tool("large_result_auto", "One-shot server-side format selection.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "rows_per_chunk", integerProp("Rows per chunk"),
                        "result_id", stringProp("Optional materialized result id"),
                        "optimization_target", stringProp("Optimization target"),
                        "prefer_streaming", booleanProp("Prefer streaming formats"),
                        "use_mab", booleanProp("Use MAB for selection")),
                List.of("n_rows", "n_cols")),
                args -> handleLargeResultAuto(args)));

        specs.add(tool("describe_and_fetch_result",
                "Fused enhanced workflow: format hints + heuristic selection + payload in one MCP call "
                        + "(same semantics as large_result_auto; eliminates the separate describe round trip).",
                objectSchema(
                        Map.of(
                                "n_rows", integerProp("Number of rows"),
                                "n_cols", integerProp("Number of columns"),
                                "rows_per_chunk", integerProp("Rows per chunk"),
                                "result_id", stringProp("Optional materialized result id"),
                                "optimization_target", stringProp("Optimization target"),
                                "prefer_streaming", booleanProp("Prefer streaming formats"),
                                "use_mab", booleanProp("Use MAB for selection")),
                        List.of("n_rows", "n_cols")),
                args -> handleDescribeAndFetchResult(args)));

        specs.add(tool("record_format_outcome", "Update server-side MAB state.", objectSchema(
                Map.of(
                        "n_rows", integerProp("Number of rows"),
                        "n_cols", integerProp("Number of columns"),
                        "optimization_target", stringProp("Optimization target"),
                        "format_used", stringProp("Format that was used"),
                        "bytes", integerProp("Observed bytes"),
                        "latency_s", numberProp("Observed latency seconds"),
                        "time_to_first_rows_s", numberProp("Observed time to first rows")),
                List.of("n_rows", "n_cols", "optimization_target", "format_used")),
                args -> handleRecordFormatOutcome(args)));

        specs.add(tool("bird_query_json", "Execute BIRD SQL and return inline JSON records.", objectSchema(
                Map.of(
                        "db_id", stringProp("BIRD database id"),
                        "sql", stringProp("SQL query"),
                        "max_rows", integerProp("Maximum rows to return")),
                List.of("db_id", "sql")),
                args -> handleBirdQueryJson(args)));

        specs.add(tool("bird_query_materialize", "Execute BIRD SQL and materialize to Parquet.", objectSchema(
                Map.of(
                        "db_id", stringProp("BIRD database id"),
                        "sql", stringProp("SQL query"),
                        "max_rows", integerProp("Maximum rows to return")),
                List.of("db_id", "sql")),
                args -> handleBirdQueryMaterialize(args)));

        specs.add(tool("bird_query_auto", "Execute BIRD SQL with one-shot format selection.", objectSchema(
                Map.of(
                        "db_id", stringProp("BIRD database id"),
                        "sql", stringProp("SQL query"),
                        "optimization_target", stringProp("Optimization target"),
                        "rows_per_chunk", integerProp("Rows per chunk"),
                        "prefer_streaming", booleanProp("Prefer streaming"),
                        "use_mab", booleanProp("Use MAB"),
                        "max_rows", integerProp("Maximum rows")),
                List.of("db_id", "sql")),
                args -> handleBirdQueryAuto(args)));

        specs.add(tool("bird_query_run_inline", "Execute BIRD SQL + select format in one round trip.", objectSchema(
                Map.of(
                        "db_id", stringProp("BIRD database id"),
                        "sql", stringProp("SQL query"),
                        "optimization_target", stringProp("Optimization target"),
                        "rows_per_chunk", integerProp("Rows per chunk"),
                        "prefer_streaming", booleanProp("Prefer streaming"),
                        "use_mab", booleanProp("Use MAB"),
                        "max_rows", integerProp("Maximum rows")),
                List.of("db_id", "sql")),
                args -> handleBirdQueryRunInline(args)));

        specs.add(tool("validate_file", "Validate a local file before use.", objectSchema(
                Map.of(
                        "file_path", stringProp("Path to local file"),
                        "expected_type", stringProp("Optional expected MIME type")),
                List.of("file_path")),
                args -> handleValidateFile(args)));

        specs.add(tool("authenticate_client", "Issue a short-lived session token.", objectSchema(
                Map.of(
                        "username", stringProp("Username"),
                        "credentials", stringProp("Credentials")),
                List.of("username", "credentials")),
                args -> handleAuthenticateClient(args)));

        specs.add(tool("access_protected_resource", "Access protected resource with JWT + RBAC.", objectSchema(
                Map.of(
                        "session_token", stringProp("Session token"),
                        "resource_path", stringProp("Resource path"),
                        "operation", stringProp("Operation (read/write)")),
                List.of("session_token", "resource_path")),
                args -> handleAccessProtectedResource(args)));

        specs.add(tool("scan_for_pii", "Detect common PII patterns.", objectSchema(
                Map.of("text_content", stringProp("Text to scan")),
                List.of("text_content")),
                args -> handleScanForPii(args)));

        specs.add(tool("encrypt_sensitive_data", "Encrypt sensitive data with AES-GCM.", objectSchema(
                Map.of("data_content", stringProp("Data to encrypt")),
                List.of("data_content")),
                args -> handleEncryptSensitiveData(args)));

        return specs;
    }

    private CallToolResult handleLargeJson(Map<String, Object> args) throws IOException {
        String resultId = strArg(args, "result_id");
        if (resultId != null) {
            ResultConfig config = runtimeState.getResultRegistry().get(resultId);
            if (config == null) {
                throw new IllegalArgumentException("Unknown result_id: " + resultId);
            }
            if (config.getCachedJsonRecords() != null) {
                payloadCache.touch(resultId);
                return toolResult("Returning cached JSON records.", Map.of("result", config.getCachedJsonRecords()));
            }
            TabularData df = tabularService.resolveDataframe(config, 0, null);
            int cells = df.numRows() * df.numCols();
            Integer cap = runtimeState.jsonCellsCap();
            if (cap != null && cells > cap) {
                throw new IllegalArgumentException("Result too large for JSON (" + cells + " cells > " + cap + ")");
            }
            return toolResult("Returning JSON records.", Map.of("result", df.toRecords()));
        }

        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        TabularData df = tabularService.generateDataframe(nRows, nCols, 0);
        return toolResult("Returning generated JSON records.", Map.of("result", df.toRecords()));
    }

    private CallToolResult handleLargeParquetBlob(Map<String, Object> args) throws IOException {
        String comp = strArg(args, "compression");
        if (comp == null) {
            comp = tabularService.getDefaultCompression();
        }
        String encStrat = strArg(args, "encoding_strategy");
        if (encStrat == null) {
            encStrat = tabularService.getDefaultEncodingStrategy();
        }

        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        Path materializedPath = null;
        String sourceId = strArg(args, "result_id");
        if (sourceId != null) {
            ResultConfig src = runtimeState.getResultRegistry().get(sourceId);
            if (src == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
            nRows = src.getNRows();
            nCols = src.getNCols();
            materializedPath = src.getMaterializedPath();
        }

        String newId = sourceId != null ? sourceId : UUID.randomUUID().toString();
        ResultConfig cfg;
        if (sourceId != null) {
            cfg = runtimeState.getResultRegistry().get(sourceId);
            if (cfg == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
        } else {
            cfg = new ResultConfig(nRows, nCols);
            cfg.setMaterializedPath(materializedPath);
            runtimeState.getResultRegistry().put(newId, cfg);
        }
        cfg.setCompression(comp);
        cfg.setEncodingStrategy(encStrat);

        Map<String, Object> descriptor = new LinkedHashMap<>();
        descriptor.put("mode", "parquet_blob");
        descriptor.put("id", newId);
        descriptor.put("url", serverBaseUrl + "/blobs/" + newId + ".parquet");
        descriptor.put("n_rows", nRows);
        descriptor.put("n_cols", nCols);
        descriptor.put("compression", comp);
        descriptor.put("encoding_strategy", encStrat);

        return toolResult(
                "Parquet blob prepared with id=" + newId + ", rows=" + nRows + ", cols=" + nCols,
                descriptor,
                Map.of("result_id", newId));
    }

    private CallToolResult handleLargeParquetStream(Map<String, Object> args) throws IOException {
        int rowsPerChunk = intArg(args, "rows_per_chunk", 0);
        if (rowsPerChunk <= 0) {
            throw new IllegalArgumentException("rows_per_chunk must be positive");
        }

        String comp = strArg(args, "compression");
        if (comp == null) {
            comp = tabularService.getDefaultCompression();
        }
        String encStrat = strArg(args, "encoding_strategy");
        if (encStrat == null) {
            encStrat = tabularService.getDefaultEncodingStrategy();
        }

        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        Path materializedPath = null;
        String sourceId = strArg(args, "result_id");
        if (sourceId != null) {
            ResultConfig src = runtimeState.getResultRegistry().get(sourceId);
            if (src == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
            nRows = src.getNRows();
            nCols = src.getNCols();
            materializedPath = src.getMaterializedPath();
        }

        String newId = sourceId != null ? sourceId : UUID.randomUUID().toString();
        ResultConfig cfg;
        if (sourceId != null) {
            cfg = runtimeState.getResultRegistry().get(sourceId);
            if (cfg == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
        } else {
            cfg = new ResultConfig(nRows, nCols);
            cfg.setMaterializedPath(materializedPath);
            runtimeState.getResultRegistry().put(newId, cfg);
        }
        cfg.setRowsPerChunk(rowsPerChunk);
        cfg.setCompression(comp);
        cfg.setEncodingStrategy(encStrat);

        Map<String, Object> descriptor = new LinkedHashMap<>();
        descriptor.put("mode", "parquet_stream");
        descriptor.put("id", newId);
        descriptor.put("url", serverBaseUrl + "/streams/" + newId);
        descriptor.put("n_rows", nRows);
        descriptor.put("n_cols", nCols);
        descriptor.put("rows_per_chunk", rowsPerChunk);
        descriptor.put("compression", comp);
        descriptor.put("encoding_strategy", encStrat);

        return toolResult("Parquet stream prepared with id=" + newId, descriptor, Map.of("result_id", newId));
    }

    private CallToolResult handleLargeArrowIpcBlob(Map<String, Object> args) throws IOException {
        String ipcComp = strArg(args, "ipc_compression");
        if (ipcComp == null) {
            ipcComp = tabularService.getDefaultArrowIpcCompression();
        }
        if (!VALID_ARROW_IPC_COMPRESSIONS.contains(ipcComp)) {
            ipcComp = "none";
        }

        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        Path materializedPath = null;
        String sourceId = strArg(args, "result_id");
        if (sourceId != null) {
            ResultConfig src = runtimeState.getResultRegistry().get(sourceId);
            if (src == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
            nRows = src.getNRows();
            nCols = src.getNCols();
            materializedPath = src.getMaterializedPath();
        }

        String newId = sourceId != null ? sourceId : UUID.randomUUID().toString();
        ResultConfig cfg;
        if (sourceId != null) {
            cfg = runtimeState.getResultRegistry().get(sourceId);
            if (cfg == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
        } else {
            cfg = new ResultConfig(nRows, nCols);
            cfg.setMaterializedPath(materializedPath);
            runtimeState.getResultRegistry().put(newId, cfg);
        }
        cfg.setIpcCompression(ipcComp);

        Map<String, Object> descriptor = new LinkedHashMap<>();
        descriptor.put("mode", "arrow_ipc_blob");
        descriptor.put("id", newId);
        descriptor.put("url", serverBaseUrl + "/ipc-blobs/" + newId + ".arrow");
        descriptor.put("n_rows", nRows);
        descriptor.put("n_cols", nCols);
        descriptor.put("ipc_compression", ipcComp);

        return toolResult("Arrow IPC blob prepared with id=" + newId, descriptor, Map.of("result_id", newId));
    }

    private CallToolResult handleLargeArrowIpcStream(Map<String, Object> args) throws IOException {
        int rowsPerChunk = intArg(args, "rows_per_chunk", 0);
        if (rowsPerChunk <= 0) {
            throw new IllegalArgumentException("rows_per_chunk must be positive");
        }

        String ipcComp = strArg(args, "ipc_compression");
        if (ipcComp == null) {
            ipcComp = tabularService.getDefaultArrowIpcCompression();
        }
        if (!VALID_ARROW_IPC_COMPRESSIONS.contains(ipcComp)) {
            ipcComp = "none";
        }

        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        Path materializedPath = null;
        String sourceId = strArg(args, "result_id");
        if (sourceId != null) {
            ResultConfig src = runtimeState.getResultRegistry().get(sourceId);
            if (src == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
            nRows = src.getNRows();
            nCols = src.getNCols();
            materializedPath = src.getMaterializedPath();
        }

        String newId = sourceId != null ? sourceId : UUID.randomUUID().toString();
        ResultConfig cfg;
        if (sourceId != null) {
            cfg = runtimeState.getResultRegistry().get(sourceId);
            if (cfg == null) {
                throw new IllegalArgumentException("Unknown result_id: " + sourceId);
            }
        } else {
            cfg = new ResultConfig(nRows, nCols);
            cfg.setMaterializedPath(materializedPath);
            runtimeState.getResultRegistry().put(newId, cfg);
        }
        cfg.setRowsPerChunk(rowsPerChunk);
        cfg.setIpcCompression(ipcComp);

        Map<String, Object> descriptor = new LinkedHashMap<>();
        descriptor.put("mode", "arrow_ipc_stream");
        descriptor.put("id", newId);
        descriptor.put("url", serverBaseUrl + "/ipc-streams/" + newId);
        descriptor.put("n_rows", nRows);
        descriptor.put("n_cols", nCols);
        descriptor.put("rows_per_chunk", rowsPerChunk);
        descriptor.put("ipc_compression", ipcComp);

        return toolResult("Arrow IPC stream prepared with id=" + newId, descriptor, Map.of("result_id", newId));
    }

    private CallToolResult handleDescribeResultFormats(Map<String, Object> args) throws IOException {
        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        int rowsPerChunk = intArg(args, "rows_per_chunk", 8192);
        String resultId = strArg(args, "result_id");
        boolean preferStreaming = boolArg(args, "prefer_streaming", false);
        OptimizationTarget selTarget = resolveTarget(strArg(args, "optimization_target"));

        if (resultId != null) {
            ResultConfig cfg = runtimeState.getResultRegistry().get(resultId);
            if (cfg != null && "unstructured".equals(cfg.getPayloadKind())) {
                return describeUnstructuredFormats(cfg, selTarget, preferStreaming);
            }
        }

        String comp = tabularService.getDefaultCompression();
        String encStrat = tabularService.getDefaultEncodingStrategy();
        String ipcComp = tabularService.getDefaultArrowIpcCompression();

        Map<String, Object> hints = tabularService.getTabularSizeHintsCached(
                nRows, nCols, rowsPerChunk, resultId, comp, encStrat, ipcComp);
        nRows = toInt(hints.getOrDefault("resolved_n_rows", nRows));
        nCols = toInt(hints.getOrDefault("resolved_n_cols", nCols));

        Map<String, Object> hintsForSelect = tabularHintsForSelect(hints);
        SelectionContext selCtx = new SelectionContext(nRows, nCols, selTarget, preferStreaming);
        String recommended = formatSelector.selectFormatWithHints(selCtx, hintsForSelect).value();

        Map<String, Object> structured = new LinkedHashMap<>();
        structured.put("approx_rows", nRows);
        structured.put("approx_cols", nCols);
        structured.put("parquet_compression", comp);
        structured.put("parquet_encoding_strategy", encStrat);
        structured.put("arrow_ipc_compression", ipcComp);
        structured.put("recommended_format", recommended);
        structured.put("recommendation_target", selTarget.value());
        structured.put("recommendation_prefer_streaming", preferStreaming);
        structured.put("formats", tabularFormatsBlock(hints));

        return toolResult("Format hints for tabular payload.", structured);
    }

    private CallToolResult handleDescribeAndFetchResult(Map<String, Object> args) throws Exception {
        return handleLargeResultAuto(args);
    }

    private CallToolResult handleLargeResultAuto(Map<String, Object> args) throws Exception {
        int nRows = intArg(args, "n_rows", 0);
        int nCols = intArg(args, "n_cols", 0);
        int rowsPerChunk = intArg(args, "rows_per_chunk", 8192);
        String resultId = strArg(args, "result_id");
        boolean preferStreaming = boolArg(args, "prefer_streaming", false);
        boolean useMab = boolArg(args, "use_mab", false);
        OptimizationTarget selTarget = resolveTarget(strArg(args, "optimization_target"));

        if (resultId != null) {
            ResultConfig cfg = runtimeState.getResultRegistry().get(resultId);
            if (cfg != null && "unstructured".equals(cfg.getPayloadKind())) {
                return largeResultAutoUnstructured(cfg, selTarget, preferStreaming, useMab);
            }
        }

        String comp = tabularService.getDefaultCompression();
        String encStrat = tabularService.getDefaultEncodingStrategy();
        String ipcComp = tabularService.getDefaultArrowIpcCompression();

        Map<String, Object> hints = tabularService.getTabularSizeHintsCached(
                nRows, nCols, rowsPerChunk, resultId, comp, encStrat, ipcComp);
        int resolvedRows = toInt(hints.getOrDefault("resolved_n_rows", nRows));
        int resolvedCols = toInt(hints.getOrDefault("resolved_n_cols", nCols));

        SelectionContext selCtx = new SelectionContext(resolvedRows, resolvedCols, selTarget, preferStreaming);
        Map<String, Object> hintsForSelect = tabularHintsForSelect(hints);
        FormatName chosen = chooseFormat(selCtx, hintsForSelect, useMab);

        return deliverChosenFormat(
                chosen, resolvedRows, resolvedCols, rowsPerChunk, resultId,
                selTarget, comp, encStrat, ipcComp);
    }

    /**
     * Deliver the chosen tabular format (inline JSON or HTTP descriptor). Shared by
     * large_result_auto, describe_and_fetch_result, and bird_query_run_inline so we
     * do not re-run hint computation or allocate a second result_id.
     */
    private CallToolResult deliverChosenFormat(
            FormatName chosen,
            int resolvedRows,
            int resolvedCols,
            int rowsPerChunk,
            String resultId,
            OptimizationTarget selTarget,
            String comp,
            String encStrat,
            String ipcComp) throws Exception {
        if (chosen == FormatName.JSON) {
            try {
                List<Map<String, Object>> records = resolveJsonRecords(resultId, resolvedRows, resolvedCols);
                Map<String, Object> structured = new LinkedHashMap<>();
                structured.put("payload_kind", "tabular");
                structured.put("chosen_format", "json");
                structured.put("optimization_target", selTarget.value());
                structured.put("payload", Map.of("kind", "json", "records", records));
                structured.put("decode", Map.of("encoding", "json_records", "transport", "inline"));
                return toolResult("Returning JSON records (json).", structured);
            } catch (Exception e) {
                chosen = FormatName.PARQUET_BLOB;
            }
        }

        Map<String, Object> delegateArgs = new HashMap<>();
        delegateArgs.put("n_rows", resolvedRows);
        delegateArgs.put("n_cols", resolvedCols);
        delegateArgs.put("rows_per_chunk", rowsPerChunk);
        delegateArgs.put("result_id", resultId);

        if (chosen == FormatName.PARQUET_STREAM) {
            delegateArgs.put("compression", comp);
            delegateArgs.put("encoding_strategy", encStrat);
            CallToolResult desc = handleLargeParquetStream(delegateArgs);
            return wrapDescriptorResult("tabular", "parquet_stream", selTarget, desc, Map.of(
                    "encoding", "parquet",
                    "transport", "http_length_prefixed_stream",
                    "url", descriptorField(desc, "url"),
                    "rows_per_chunk", descriptorField(desc, "rows_per_chunk")));
        }

        if (chosen == FormatName.ARROW_IPC_STREAM) {
            delegateArgs.put("ipc_compression", ipcComp);
            CallToolResult desc = handleLargeArrowIpcStream(delegateArgs);
            return wrapDescriptorResult("tabular", "arrow_ipc_stream", selTarget, desc, Map.of(
                    "encoding", "arrow_ipc",
                    "transport", "http_length_prefixed_stream",
                    "url", descriptorField(desc, "url"),
                    "rows_per_chunk", descriptorField(desc, "rows_per_chunk")));
        }

        if (chosen == FormatName.ARROW_IPC_BLOB) {
            delegateArgs.put("ipc_compression", ipcComp);
            CallToolResult desc = handleLargeArrowIpcBlob(delegateArgs);
            return wrapDescriptorResult("tabular", "arrow_ipc_blob", selTarget, desc, Map.of(
                    "encoding", "arrow_ipc", "transport", "http_blob", "url", descriptorField(desc, "url")));
        }

        delegateArgs.put("compression", comp);
        delegateArgs.put("encoding_strategy", encStrat);
        CallToolResult desc = handleLargeParquetBlob(delegateArgs);
        return wrapDescriptorResult("tabular", "parquet_blob", selTarget, desc, Map.of(
                "encoding", "parquet", "transport", "http_blob", "url", descriptorField(desc, "url")));
    }

    private CallToolResult handleRecordFormatOutcome(Map<String, Object> args) throws IOException {
        OptimizationTarget target = resolveTarget(strArg(args, "optimization_target"));
        SelectionContext ctx = new SelectionContext(
                intArg(args, "n_rows", 0), intArg(args, "n_cols", 0), target);
        Map<String, Object> outcome = new HashMap<>();
        outcome.put("bytes", args.get("bytes"));
        outcome.put("latency_s", args.get("latency_s"));
        outcome.put("time_to_first_rows_s", args.get("time_to_first_rows_s"));

        FormatName formatUsed = FormatName.fromString(strArg(args, "format_used"));
        var mabState = formatMab.loadMabState(runtimeState.getServerMabStatePath());
        formatMab.recordOutcome(ctx, formatUsed, outcome, mabState, FormatMab.DEFAULT_HISTORY_PATH);
        formatMab.saveMabState(mabState, runtimeState.getServerMabStatePath());

        return toolResult("Outcome recorded to server MAB state.", Map.of("recorded", true));
    }

    private CallToolResult handleBirdQueryJson(Map<String, Object> args) throws SQLException {
        List<Map<String, Object>> records = birdQueryService.queryJson(
                strArg(args, "db_id"), strArg(args, "sql"), intArg(args, "max_rows", 500_000));
        return toolResult("BIRD query returned " + records.size() + " records.", Map.of("result", records));
    }

    private CallToolResult handleBirdQueryMaterialize(Map<String, Object> args) throws Exception {
        Map<String, Object> mat = birdQueryService.materialize(
                strArg(args, "db_id"), strArg(args, "sql"), intArg(args, "max_rows", 500_000));
        return toolResult("BIRD query materialized.", mat);
    }

    private CallToolResult handleBirdQueryAuto(Map<String, Object> args) throws Exception {
        Map<String, Object> mat = birdQueryService.materialize(
                strArg(args, "db_id"), strArg(args, "sql"), intArg(args, "max_rows", 500_000));
        String rid = String.valueOf(mat.get("result_id"));
        Map<String, Object> forwarded = new HashMap<>();
        forwarded.put("n_rows", mat.get("n_rows"));
        forwarded.put("n_cols", mat.get("n_cols"));
        forwarded.put("rows_per_chunk", intArg(args, "rows_per_chunk", 8192));
        forwarded.put("result_id", rid);
        String optimizationTarget = strArg(args, "optimization_target");
        if (optimizationTarget != null) {
            forwarded.put("optimization_target", optimizationTarget);
        }
        forwarded.put("prefer_streaming", boolArg(args, "prefer_streaming", false));
        forwarded.put("use_mab", boolArg(args, "use_mab", false));
        return handleLargeResultAuto(forwarded);
    }

    private CallToolResult handleBirdQueryRunInline(Map<String, Object> args) throws Exception {
        Path dbPath = birdQueryService.resolveBirdSqlitePath(strArg(args, "db_id"));
        if (dbPath == null) {
            throw new IllegalArgumentException("Could not resolve BIRD sqlite for db_id. Set BIRD_SQLITE_ROOT.");
        }

        TabularData df = birdQueryService.executeSqliteQuery(
                dbPath, strArg(args, "sql"), intArg(args, "max_rows", 500_000));
        if (df.numRows() == 0) {
            throw new IllegalArgumentException("Empty result");
        }

        int nRows = df.numRows();
        int nCols = df.numCols();
        int rowsPerChunk = intArg(args, "rows_per_chunk", 8192);
        boolean preferStreaming = boolArg(args, "prefer_streaming", false);
        boolean useMab = boolArg(args, "use_mab", false);
        OptimizationTarget selTarget = resolveTarget(strArg(args, "optimization_target"));

        String comp = tabularService.getDefaultCompression();
        String encStrat = tabularService.getDefaultEncodingStrategy();
        String ipcComp = tabularService.getDefaultArrowIpcCompression();

        ResultConfig scratch = new ResultConfig(nRows, nCols);
        scratch.setCachedDataframe(df);
        VectorSchemaRoot table = tabularService.resolveArrowTable(scratch);

        Map<String, Object> hints = tabularService.computeTabularSizeHintsFromDf(
                df, rowsPerChunk, comp, encStrat, ipcComp, table);
        int resolvedRows = toInt(hints.getOrDefault("resolved_n_rows", nRows));
        int resolvedCols = toInt(hints.getOrDefault("resolved_n_cols", nCols));

        SelectionContext selCtx = new SelectionContext(resolvedRows, resolvedCols, selTarget, preferStreaming);
        FormatName chosen = chooseFormat(selCtx, tabularHintsForSelect(hints), useMab);

        if (chosen == FormatName.JSON) {
            try {
                Integer cap = runtimeState.jsonCellsCap();
                int cells = resolvedRows * resolvedCols;
                if (cap != null && cells > cap) {
                    throw new IllegalArgumentException("Result too large for JSON");
                }
                Map<String, Object> structured = new LinkedHashMap<>();
                structured.put("payload_kind", "tabular");
                structured.put("chosen_format", "json");
                structured.put("optimization_target", selTarget.value());
                structured.put("payload", Map.of("kind", "json", "records", df.toRecords()));
                structured.put("decode", Map.of("encoding", "json_records", "transport", "inline"));
                return toolResult("Returning JSON records (json).", structured);
            } catch (Exception e) {
                chosen = FormatName.PARQUET_BLOB;
            }
        }

        byte[] pqBytes = tabularService.encodeParquet(table, comp, encStrat);
        String mid = UUID.randomUUID().toString();
        Path path = runtimeState.getRoot().resolve("results/materialized/bird_inline_" + mid + ".parquet");
        Files.createDirectories(path.getParent());
        Files.write(path, pqBytes);

        ResultConfig cfg = new ResultConfig(nRows, nCols);
        cfg.setCompression(comp);
        cfg.setEncodingStrategy(encStrat);
        cfg.setMaterializedPath(path);
        try {
            tabularService.populateMaterializedCaches(cfg, mid, df, table, pqBytes, rowsPerChunk);
        } catch (Exception e) {
            cfg.setCachedHints(null);
        }
        runtimeState.getResultRegistry().put(mid, cfg);

        return deliverChosenFormat(
                chosen, resolvedRows, resolvedCols, rowsPerChunk, mid,
                selTarget, comp, encStrat, ipcComp);
    }

    private CallToolResult handleValidateFile(Map<String, Object> args) {
        Map<String, Object> descriptor = fileValidationService.validateAndRegister(
                strArg(args, "file_path"), strArg(args, "expected_type"));
        if (!Boolean.TRUE.equals(descriptor.get("valid"))) {
            return toolResult("File validation failed: " + descriptor.get("error"), descriptor);
        }
        return toolResult("File validation passed: " + descriptor.get("details"), descriptor);
    }

    private CallToolResult handleAuthenticateClient(Map<String, Object> args) {
        Map<String, Object> result = authService.authenticate(
                strArg(args, "username"), strArg(args, "credentials"));
        if (!Boolean.TRUE.equals(result.get("authenticated"))) {
            return toolResult("Authentication failed", result);
        }
        return toolResult("Authentication successful. Session token issued.", result);
    }

    private CallToolResult handleAccessProtectedResource(Map<String, Object> args) {
        Map<String, Object> result = authService.accessProtectedResource(
                strArg(args, "session_token"),
                strArg(args, "resource_path"),
                strArg(args, "operation") != null ? strArg(args, "operation") : "read");
        if (!Boolean.TRUE.equals(result.get("authorized"))) {
            return toolResult("Access denied.", result);
        }
        return toolResult("Access granted.", result);
    }

    private CallToolResult handleScanForPii(Map<String, Object> args) {
        Map<String, Object> result = privacyService.scanForPii(strArg(args, "text_content"));
        if (!Boolean.TRUE.equals(result.get("pii_found"))) {
            return toolResult("No PII detected in content", result);
        }
        return toolResult("PII detected and sanitized.", result);
    }

    private CallToolResult handleEncryptSensitiveData(Map<String, Object> args) {
        Map<String, Object> result = privacyService.encryptSensitiveData(strArg(args, "data_content"));
        return toolResult("Data encrypted successfully.", result);
    }

    private CallToolResult describeUnstructuredFormats(
            ResultConfig cfg, OptimizationTarget selTarget, boolean preferStreaming) throws IOException {
        if (cfg.getRawPath() == null || !Files.isRegularFile(cfg.getRawPath())) {
            throw new IllegalArgumentException("Unstructured result_id missing file");
        }
        long rawBytes = Files.size(cfg.getRawPath());
        Long gzipBytes = null;
        if (cfg.getRawGzipPath() != null && Files.isRegularFile(cfg.getRawGzipPath())) {
            gzipBytes = Files.size(cfg.getRawGzipPath());
        }
        Long inlineBytes = rawBytes <= RuntimeState.MAX_INLINE_TEXT_BYTES ? rawBytes : null;

        Map<String, Object> hintsForSelect = new HashMap<>();
        hintsForSelect.put("raw_bytes", (int) rawBytes);
        hintsForSelect.put("gzip_bytes", gzipBytes != null ? gzipBytes.intValue() : null);
        hintsForSelect.put("text_inline_bytes", inlineBytes != null ? inlineBytes.intValue() : null);

        SelectionContext selCtx = new SelectionContext(0, 0, selTarget, preferStreaming);
        String recommended = formatSelector.selectFormatWithHints(selCtx, hintsForSelect).value();

        Map<String, Object> structured = new LinkedHashMap<>();
        structured.put("payload_kind", "unstructured");
        structured.put("mime_type", cfg.getRawMimeType() != null ? cfg.getRawMimeType() : "application/octet-stream");
        structured.put("recommended_format", recommended);
        structured.put("recommendation_target", selTarget.value());
        Map<String, Object> gzipFormat = new LinkedHashMap<>();
        gzipFormat.put("supported", true);
        gzipFormat.put("approx_bytes", gzipBytes);
        Map<String, Object> textInlineFormat = new LinkedHashMap<>();
        textInlineFormat.put("supported", inlineBytes != null);
        textInlineFormat.put("approx_bytes", inlineBytes);
        Map<String, Object> formats = new LinkedHashMap<>();
        formats.put("raw_blob", Map.of("supported", true, "approx_bytes", rawBytes));
        formats.put("gzip_blob", gzipFormat);
        formats.put("text_inline", textInlineFormat);
        structured.put("formats", formats);

        return toolResult("Unstructured format hints.", structured);
    }

    private CallToolResult largeResultAutoUnstructured(
            ResultConfig cfg, OptimizationTarget selTarget, boolean preferStreaming, boolean useMab)
            throws IOException {
        if (cfg.getRawPath() == null || !Files.isRegularFile(cfg.getRawPath())) {
            throw new IllegalArgumentException("Unstructured result_id missing file");
        }
        long rawBytes = Files.size(cfg.getRawPath());
        Long gzipBytes = cfg.getRawGzipPath() != null && Files.isRegularFile(cfg.getRawGzipPath())
                ? Files.size(cfg.getRawGzipPath())
                : null;
        Long inlineBytes = rawBytes <= RuntimeState.MAX_INLINE_TEXT_BYTES ? rawBytes : null;

        SelectionContext selCtx = new SelectionContext(0, 0, selTarget, preferStreaming);
        Map<String, Object> hints = new HashMap<>();
        hints.put("raw_bytes", (int) rawBytes);
        hints.put("gzip_bytes", gzipBytes != null ? gzipBytes.intValue() : null);
        hints.put("text_inline_bytes", inlineBytes != null ? inlineBytes.intValue() : null);
        FormatName chosen = chooseFormat(selCtx, hints, useMab);

        if (chosen == FormatName.TEXT_INLINE) {
            byte[] data = Files.readAllBytes(cfg.getRawPath());
            try {
                String text = new String(data, cfg.getRawCharset() != null ? cfg.getRawCharset() : "UTF-8");
                Map<String, Object> structured = new LinkedHashMap<>();
                structured.put("payload_kind", "unstructured");
                structured.put("chosen_format", "text_inline");
                structured.put("optimization_target", selTarget.value());
                structured.put("payload", Map.of("kind", "text", "text", text));
                structured.put("decode", Map.of(
                        "encoding", "text",
                        "transport", "inline",
                        "mime_type", cfg.getRawMimeType() != null ? cfg.getRawMimeType() : "text/plain",
                        "charset", cfg.getRawCharset() != null ? cfg.getRawCharset() : "utf-8"));
                return toolResult("Returning inline text payload.", structured);
            } catch (Exception e) {
                chosen = FormatName.RAW_BLOB;
            }
        }

        if (chosen == FormatName.GZIP_BLOB) {
            return unstructuredDescriptorResult(cfg, selTarget, "gzip_blob", "gzip_blob", "/raw-gzip/", Map.of(
                    "encoding", "raw_bytes",
                    "transport", "http_blob",
                    "content_encoding", "gzip"));
        }

        return unstructuredDescriptorResult(cfg, selTarget, "raw_blob", "raw_blob", "/raw/", Map.of(
                "encoding", "raw_bytes", "transport", "http_blob"));
    }

    private CallToolResult unstructuredDescriptorResult(
            ResultConfig cfg,
            OptimizationTarget selTarget,
            String mode,
            String chosenFormat,
            String urlPrefix,
            Map<String, Object> decodeExtra)
            throws IOException {
        String newId = UUID.randomUUID().toString();
        ResultConfig copy = new ResultConfig(0, 0);
        copy.setPayloadKind("unstructured");
        copy.setRawPath(cfg.getRawPath());
        copy.setRawMimeType(cfg.getRawMimeType());
        copy.setRawCharset(cfg.getRawCharset());
        copy.setRawGzipPath(cfg.getRawGzipPath());
        runtimeState.getResultRegistry().put(newId, copy);

        Map<String, Object> descriptor = new LinkedHashMap<>();
        descriptor.put("mode", mode);
        descriptor.put("id", newId);
        descriptor.put("url", serverBaseUrl + urlPrefix + newId);
        descriptor.put("mime_type", cfg.getRawMimeType() != null ? cfg.getRawMimeType() : "application/octet-stream");

        Map<String, Object> decode = new LinkedHashMap<>(decodeExtra);
        decode.put("url", descriptor.get("url"));
        decode.put("mime_type", descriptor.get("mime_type"));

        Map<String, Object> structured = new LinkedHashMap<>();
        structured.put("payload_kind", "unstructured");
        structured.put("chosen_format", chosenFormat);
        structured.put("optimization_target", selTarget.value());
        structured.put("payload", mergeMaps(Map.of("kind", "descriptor"), descriptor));
        structured.put("decode", decode);

        return toolResult("Returning " + chosenFormat + " descriptor.", structured);
    }

    private FormatName chooseFormat(
            SelectionContext selCtx, Map<String, Object> hintsForSelect, boolean useMab) {
        if (useMab) {
            var mabState = formatMab.loadMabState(runtimeState.getServerMabStatePath());
            return formatMab.selectFormatWithMab(selCtx, hintsForSelect, mabState, null);
        }
        return formatSelector.selectFormatWithHints(selCtx, hintsForSelect);
    }

    private CallToolResult wrapDescriptorResult(
            String payloadKind,
            String chosenFormat,
            OptimizationTarget selTarget,
            CallToolResult desc,
            Map<String, Object> decode) {
        Map<String, Object> structured = new LinkedHashMap<>();
        structured.put("payload_kind", payloadKind);
        structured.put("chosen_format", chosenFormat);
        structured.put("optimization_target", selTarget.value());
        @SuppressWarnings("unchecked")
        Map<String, Object> descriptorMap = (Map<String, Object>) desc.structuredContent();
        structured.put("payload", mergeMaps(Map.of("kind", "descriptor"), descriptorMap));
        structured.put("decode", decode);
        return toolResult("Returning " + chosenFormat + " descriptor.", structured);
    }

    private static Map<String, Object> tabularHintsForSelect(Map<String, Object> hints) {
        Map<String, Object> out = new HashMap<>();
        out.put("json_bytes", toInt(hints.get("json_bytes")));
        out.put("parquet_bytes", toInt(hints.get("parquet_bytes")));
        out.put("parquet_stream_first_chunk_bytes", toInt(hints.get("parquet_stream_first_chunk_bytes")));
        out.put("arrow_ipc_bytes", toInt(hints.get("arrow_ipc_bytes")));
        out.put("arrow_ipc_stream_first_chunk_bytes", toInt(hints.get("arrow_ipc_stream_first_chunk_bytes")));
        return out;
    }

    private static Map<String, Object> tabularFormatsBlock(Map<String, Object> hints) {
        return Map.of(
                "json", Map.of("supported", true, "approx_bytes", toInt(hints.get("json_bytes"))),
                "parquet_blob", Map.of("supported", true, "approx_bytes", toInt(hints.get("parquet_bytes"))),
                "parquet_stream", Map.of(
                        "supported", true,
                        "approx_bytes", toInt(hints.get("parquet_bytes")),
                        "approx_first_chunk_bytes", toInt(hints.get("parquet_stream_first_chunk_bytes"))),
                "arrow_ipc_blob", Map.of("supported", true, "approx_bytes", toInt(hints.get("arrow_ipc_bytes"))),
                "arrow_ipc_stream", Map.of(
                        "supported", true,
                        "approx_bytes", toInt(hints.get("arrow_ipc_bytes")),
                        "approx_first_chunk_bytes", toInt(hints.get("arrow_ipc_stream_first_chunk_bytes"))));
    }

    private OptimizationTarget resolveTarget(String raw) {
        if (raw == null || raw.isBlank()) {
            return formatSelector.getDefaultTarget();
        }
        return OptimizationTarget.fromString(raw);
    }

    @FunctionalInterface
    private interface ToolHandler {
        CallToolResult apply(Map<String, Object> arguments) throws Exception;
    }

    private static SyncToolSpecification tool(
            String name, String description, JsonSchema inputSchema, ToolHandler handler) {
        BiFunction<McpSyncServerExchange, CallToolRequest, CallToolResult> callHandler = (exchange, request) -> {
            try {
                return handler.apply(request.arguments());
            } catch (Exception e) {
                String message = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                return CallToolResult.builder()
                        .isError(true)
                        .content(List.of(new TextContent(null, message)))
                        .structuredContent(Map.of("error", message))
                        .build();
            }
        };
        return SyncToolSpecification.builder()
                .tool(Tool.builder().name(name).description(description).inputSchema(inputSchema).build())
                .callHandler(callHandler)
                .build();
    }

    private static CallToolResult toolResult(String text, Map<String, Object> structured) {
        return toolResult(text, (Object) structured, null);
    }

    private static CallToolResult toolResult(String text, Object structured) {
        return toolResult(text, structured, null);
    }

    private static CallToolResult toolResult(String text, Object structured, Map<String, Object> meta) {
        CallToolResult.Builder builder = CallToolResult.builder()
                .content(List.of(new TextContent(null, text)))
                .structuredContent(structured);
        if (meta != null) {
            builder.meta(meta);
        }
        return builder.build();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> resolveJsonRecords(
            String resultId, int resolvedRows, int resolvedCols) throws IOException {
        if (resultId != null) {
            ResultConfig config = runtimeState.getResultRegistry().get(resultId);
            if (config == null) {
                throw new IllegalArgumentException("Unknown result_id: " + resultId);
            }
            if (config.getCachedJsonRecords() != null) {
                payloadCache.touch(resultId);
                return config.getCachedJsonRecords();
            }
            TabularData df = tabularService.resolveDataframe(config, 0, null);
            int cells = df.numRows() * df.numCols();
            Integer cap = runtimeState.jsonCellsCap();
            if (cap != null && cells > cap) {
                throw new IllegalArgumentException(
                        "Result too large for JSON (" + cells + " cells > " + cap + ")");
            }
            return df.toRecords();
        }
        TabularData df = tabularService.generateDataframe(resolvedRows, resolvedCols, 0);
        return df.toRecords();
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> jsonRecords(Object structuredContent) {
        Object unwrapped = structuredContent;
        if (unwrapped instanceof Map<?, ?> map && map.containsKey("result")) {
            unwrapped = map.get("result");
        }
        if (unwrapped instanceof List<?> list) {
            return (List<Map<String, Object>>) list;
        }
        if (unwrapped instanceof Map<?, ?> map && map.get("records") instanceof List<?> records) {
            return (List<Map<String, Object>>) records;
        }
        throw new IllegalStateException("Unexpected JSON tool structured content: " + unwrapped);
    }

    private static JsonSchema objectSchema(Map<String, Object> properties, List<String> required) {
        return new JsonSchema("object", properties, required, false, null, null);
    }

    private static Map<String, Object> stringProp(String description) {
        return Map.of("type", "string", "description", description);
    }

    private static Map<String, Object> integerProp(String description) {
        return Map.of("type", "integer", "description", description);
    }

    private static Map<String, Object> numberProp(String description) {
        return Map.of("type", "number", "description", description);
    }

    private static Map<String, Object> booleanProp(String description) {
        return Map.of("type", "boolean", "description", description);
    }

    private static String strArg(Map<String, Object> args, String key) {
        Object value = args.get(key);
        if (value == null) {
            return null;
        }
        String s = String.valueOf(value);
        return s.isBlank() ? null : s;
    }

    private static int intArg(Map<String, Object> args, String key, int defaultValue) {
        Object value = args.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(String.valueOf(value));
    }

    private static boolean boolArg(Map<String, Object> args, String key, boolean defaultValue) {
        Object value = args.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Boolean b) {
            return b;
        }
        return Boolean.parseBoolean(String.valueOf(value));
    }

    private static int toInt(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(String.valueOf(value));
    }

    @SafeVarargs
    private static Map<String, Object> mergeMaps(Map<String, Object>... maps) {
        Map<String, Object> merged = new LinkedHashMap<>();
        for (Map<String, Object> map : maps) {
            merged.putAll(map);
        }
        return merged;
    }

    @SuppressWarnings("unchecked")
    private static Object descriptorField(CallToolResult desc, String key) {
        Object structured = desc.structuredContent();
        if (structured instanceof Map<?, ?> map) {
            return map.get(key);
        }
        return null;
    }

    private static String stripTrailingSlash(String url) {
        if (url == null || url.isBlank()) {
            return "http://localhost:8000";
        }
        return url.endsWith("/") ? url.substring(0, url.length() - 1) : url;
    }
}
