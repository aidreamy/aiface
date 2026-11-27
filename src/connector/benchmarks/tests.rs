use super::*;
use crate::connector::test::common::{MockResponse, mock_server, mock_server_sequence};
use serde_json::json;

const MODELS: &str = r#"{
    "models": {
        "anthropic/claude-opus-5-5": {"name": "Claude Opus 5.5"},
        "openai/gpt-6-astra": {"name": "GPT-6 Astra"},
        "openai/gpt-5.5": {"name": "GPT-5.5"},
        "deepseek/deepseek-v4-pro": {"name": "DeepSeek V4 Pro"},
        "google/gemini-3.8-flash": {"name": "Gemini 3.8 Flash"},
        "mistral/mistral-large-4": {"name": "Mistral Large 4"},
        "first/shared": {"name": "Shared"},
        "second/shared": {"name": "Shared"}
    },
    "providers": {
        "openai": {"models": {"gpt-5.5": {}}},
        "deepinfra": {"models": {
            "deepseek-ai/DeepSeek-V4-Pro": {"canonical_model_id": "deepseek/deepseek-v4-pro"}
        }},
        "openrouter": {"models": {
            "anthropic/claude-opus-5.5": {"canonical_model_id": "anthropic/claude-opus-5-5"},
            "openai/gpt-6-astra-pro": {"canonical_model_id": "openai/gpt-6-astra"},
            "mistralai/mistral-large-4-0": {"canonical_model_id": "mistral/mistral-large-4"}
        }},
        "github-copilot": {"models": {
            "claude-opus-5.5": {"canonical_model_id": "anthropic/claude-opus-5-5"}
        }}
    }
}"#;

fn terminal() -> String {
    let row = json!({
        "id": "24f711aa-9398-4033-bbac-5fc773907e11",
        "status": "display",
        "updated_at": "2026-11-09T14:29:00Z",
        "n_trials": 330,
        "metadata": {
            "model_display": {"label": "Opus 5.5"},
            "model_org": {"label": "Anthropic"},
            "agent_display": {"label": "Claude Code"},
            "reasoning_effort": "max"
        },
        "metrics": {"accuracy": 65.8, "accuracy_ci95_half_width": 3.1}
    });
    let mut hidden = row.clone();
    hidden["status"] = "hide".into();
    let mut wrong = row.clone();
    wrong["metrics"]["accuracy"] = 120.0.into();
    let mut unresolved = row.clone();
    unresolved["metadata"]["model_display"]["label"] = "Shared".into();
    unresolved["metadata"]["model_org"]["label"] = "Other".into();
    let mut measured_route = row.clone();
    measured_route["id"] = "5e0a7998-58e3-4176-83cf-f1fa15280203".into();
    measured_route["metadata"]["model_org"]["label"] = "Mistral AI".into();
    measured_route["metadata"]["model_display"] = json!({
        "label": "Mistral Large 4",
        "url": "https://openrouter.ai/mistralai/mistral-large-4-0"
    });
    json!({
        "leaderboard": {"name": "4-1-0", "package": "terminal-bench/terminal-bench"},
        "rows": [row, hidden, wrong, unresolved, measured_route, {"invalid":true}]
    })
    .to_string()
}

const DEEP: &str = r#"{
    "generated_at": "2026-09-22T06:27:15Z",
    "rows": [
        {"model":"gpt-6-astra", "provider":"openai", "harness":"mini-swe-agent",
         "pass_at_1":0.741, "pass_at_4":0.805, "ci_lo":0.712, "ci_hi":0.77,
         "reasoning_effort":"xhigh", "n_attempted":452},
        {"model":"gpt-6-astra", "provider":"openai", "harness":"mini-swe-agent",
         "pass_at_1":0.67, "reasoning_effort":"high"},
        {"model":"gemini-3-8-flash", "harness":"mini-swe-agent", "pass_at_1":0.737},
        {"model":"gpt-5-5", "harness":"mini-swe-agent", "pass_at_1":0.65},
        {"model":"deepseek-v4-pro", "harness":"mini-swe-agent", "pass_at_1":0.63},
        {"model":"new-model", "harness":"mini-swe-agent", "pass_at_1":0.0}
    ]
}"#;

#[tokio::test]
async fn cached_results_match_model_aliases_without_collapsing_variants() {
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(&dir.path().join("benchmark-models.json"), MODELS.as_bytes()).unwrap();
    write_snapshot(
        &dir.path().join("terminal-bench-4.1.json"),
        terminal().as_bytes(),
    )
    .unwrap();
    write_snapshot(&dir.path().join("deep-swe-1.1.json"), DEEP.as_bytes()).unwrap();
    let catalog = BenchmarkCatalog::load_cached(dir.path().to_str().unwrap()).await;

    let opus = catalog.for_model(Some("openrouter"), "anthropic/claude-opus-5.5");
    assert_eq!(opus, catalog.for_model(Some("claude"), "claude-opus-5-5"));
    assert_eq!(opus, catalog.for_model(Some("copilot"), "claude-opus-5.5"));
    assert_eq!(opus, catalog.for_model(None, "anthropic/claude-opus-5-5"));
    assert_eq!(opus.len(), 1);
    let row = opus[0];
    assert_eq!(row.benchmark, Benchmark::TerminalBench);
    assert!((row.success_rate - 0.658).abs() < 1e-10);
    let confidence = row.interval.unwrap();
    assert!((confidence.lower - 0.627).abs() < 1e-10);
    assert!((confidence.upper - 0.689).abs() < 1e-10);
    assert_eq!(row.harness, "Claude Code");
    assert_eq!(row.reasoning_effort.as_deref(), Some("max"));
    assert_eq!(row.attempts, Some(330));
    assert!(row.source_url.contains("/rows/24f711aa"));

    assert!(
        catalog
            .for_model(Some("openrouter"), "openai/gpt-6-astra-pro")
            .is_empty()
    );
    assert!(
        catalog
            .for_model(Some("openai"), "gpt-6-astra-2026-09-01")
            .is_empty()
    );
    assert!(catalog.for_model(Some("custom"), "gpt-6-astra").is_empty());
    assert!(catalog.for_model(None, "shared").is_empty());
    let astra = catalog.for_model(Some("openai"), "gpt-6-astra");
    assert_eq!(astra.len(), 2);
    assert_eq!(astra[0].reasoning_effort.as_deref(), Some("xhigh"));
    assert_eq!(astra[1].reasoning_effort.as_deref(), Some("high"));
    assert_eq!(astra[0].success_rate, 0.741);
    assert_eq!(
        astra[0].published_at.as_deref(),
        Some("2026-09-22T06:27:15Z")
    );
    assert_eq!(
        catalog.for_model(Some("gemini"), "gemini-3.8-flash")[0].success_rate,
        0.737
    );
    assert_eq!(
        catalog.for_model(Some("openai"), "gpt-5.5")[0].success_rate,
        0.65
    );
    assert_eq!(
        catalog.for_model(Some("deepinfra"), "deepseek-ai/DeepSeek-V4-Pro")[0].success_rate,
        0.63
    );
    let mistral = catalog.for_model(Some("openrouter"), "mistralai/mistral-large-4-0");
    assert_eq!(mistral, catalog.for_model(None, "mistral/mistral-large-4"));
    assert_eq!(
        mistral[0].reported_model_route.as_deref(),
        Some("openrouter/mistralai/mistral-large-4-0")
    );
    assert_eq!(catalog.results().len(), 9);
    assert!(
        catalog
            .results()
            .iter()
            .any(|row| row.reported_model == "Shared" && row.model_id.is_none())
    );
    assert!(
        catalog
            .results()
            .iter()
            .any(|row| row.reported_model == "new-model" && row.success_rate == 0.0)
    );
    assert!(matches!(
        catalog.status(Benchmark::TerminalBench),
        BenchmarkStatus::Cached { stale: false, .. }
    ));
}

#[tokio::test]
async fn an_unavailable_feed_and_identity_catalog_leave_direct_results_usable() {
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(&dir.path().join("deep-swe-1.1.json"), DEEP.as_bytes()).unwrap();
    let catalog = BenchmarkCatalog::load_from(Some(dir.path().into()), false).await;
    assert_eq!(
        catalog.status(Benchmark::TerminalBench),
        BenchmarkStatus::Unavailable
    );
    assert_eq!(catalog.for_model(None, "openai/gpt-6-astra").len(), 2);
    assert_eq!(catalog.identity_status(), BenchmarkStatus::Unavailable);
    assert_eq!(catalog.results().len(), 6);
}

#[test]
fn incomplete_or_wrong_terminal_leaderboard_cannot_replace_a_cache() {
    let mut board: serde_json::Value = serde_json::from_str(&terminal()).unwrap();
    board["pagination"] = json!({"total_pages":2});
    assert!(parse_terminal(board.to_string().as_bytes()).is_none());
    board["pagination"] = serde_json::Value::Null;
    board["leaderboard"]["name"] = "4-0-0".into();
    assert!(parse_terminal(board.to_string().as_bytes()).is_none());
    assert!(parse_deep_swe(br#"{"generated_at":"today","rows":[{"pass_at_4":0.9}]}"#).is_none());
}

#[test]
fn an_empty_or_hidden_leaderboard_is_a_valid_snapshot() {
    let mut board: serde_json::Value = serde_json::from_str(&terminal()).unwrap();
    board["rows"] = json!([{"status": "hide"}]);
    assert!(
        parse_terminal(board.to_string().as_bytes())
            .unwrap()
            .is_empty()
    );
    board["rows"] = json!([]);
    assert!(
        parse_terminal(board.to_string().as_bytes())
            .unwrap()
            .is_empty()
    );
    board["rows"] = json!([{"status": "display"}]);
    assert!(parse_terminal(board.to_string().as_bytes()).is_none());
    assert!(
        parse_deep_swe(br#"{"generated_at":"today","rows":[]}"#)
            .unwrap()
            .is_empty()
    );
}

fn expire(path: &Path) {
    let yesterday = SystemTime::now() - CACHE_MAX_AGE - Duration::from_secs(1);
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(yesterday)
        .unwrap();
}

#[tokio::test]
async fn expired_cache_survives_invalid_payload_and_http_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("terminal.json");
    let valid = terminal();
    write_snapshot(&path, valid.as_bytes()).unwrap();
    expire(&path);

    for (body, status) in [("{}", 200), ("temporarily unavailable", 503)] {
        let (port, _, _, server) = mock_server(body, status);
        let request = shared_client().get(format!("http://127.0.0.1:{port}/"));
        let snapshot = load_document(
            Some(path.clone()),
            Some(request),
            FEED_LIMIT,
            parse_terminal,
        )
        .await;
        server.join().unwrap();
        assert!(matches!(
            snapshot.status,
            BenchmarkStatus::Cached { stale: true, .. }
        ));
        assert_eq!(snapshot.data.unwrap().len(), 3);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), valid);
    }
}

#[tokio::test]
async fn failed_refresh_uses_snapshot_written_by_concurrent_refresh() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deep-swe.json");
    write_snapshot(&path, DEEP.as_bytes()).unwrap();
    expire(&path);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let request = shared_client().get(format!("http://{}/", listener.local_addr().unwrap()));
    let server_path = path.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut stream = BufReader::new(stream);
        let mut line = String::new();
        while stream.read_line(&mut line).await.unwrap() > 0 && line != "\r\n" {
            line.clear();
        }
        write_snapshot(&server_path, br#"{"generated_at":"today","rows":[]}"#).unwrap();
        stream.get_mut().write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
    });
    let snapshot = load_document(Some(path), Some(request), FEED_LIMIT, parse_deep_swe).await;
    server.await.unwrap();
    assert!(matches!(
        snapshot.status,
        BenchmarkStatus::Cached { stale: false, .. }
    ));
    assert!(snapshot.data.unwrap().is_empty());
}

#[cfg(unix)]
#[test]
fn reading_cache_skips_a_named_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deep-swe.json");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .unwrap()
            .success()
    );
    let (send, recv) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        send.send(read_snapshot(&path, FEED_LIMIT, parse_deep_swe).is_none())
            .unwrap();
    });
    assert!(recv.recv_timeout(Duration::from_secs(2)).unwrap());
    worker.join().unwrap();
}

#[tokio::test]
async fn network_snapshot_is_cached_and_reused_within_its_lifetime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deep-swe.json");
    let (port, _, _, server) = mock_server(DEEP, 200);
    let request = shared_client().get(format!("http://127.0.0.1:{port}/"));
    let snapshot = load_document(
        Some(path.clone()),
        Some(request),
        FEED_LIMIT,
        parse_deep_swe,
    )
    .await;
    server.join().unwrap();
    assert!(matches!(snapshot.status, BenchmarkStatus::Fetched { .. }));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), DEEP);

    let (port, calls, server) = mock_server_sequence(vec![MockResponse::new(503, "unused")]);
    let request = shared_client().get(format!("http://127.0.0.1:{port}/"));
    let snapshot = load_document(Some(path), Some(request), FEED_LIMIT, parse_deep_swe).await;
    server.join().unwrap();
    assert_eq!(*calls.lock().unwrap(), 0);
    assert!(matches!(
        snapshot.status,
        BenchmarkStatus::Cached { stale: false, .. }
    ));
}

#[tokio::test]
async fn oversized_response_is_rejected() {
    let (port, _, _, server) = mock_server(DEEP, 200);
    let request = shared_client().get(format!("http://127.0.0.1:{port}/"));
    let response = download(request, 64).await;
    server.join().unwrap();
    assert!(matches!(response, Err(DownloadError::TooLarge)));
}

#[tokio::test]
#[ignore = "requires public benchmark feeds"]
async fn live_feeds_resolve_known_models() {
    let catalog = BenchmarkCatalog::load(None).await;
    for benchmark in [Benchmark::TerminalBench, Benchmark::DeepSwe] {
        assert!(matches!(
            catalog.status(benchmark),
            BenchmarkStatus::Fetched { .. }
        ));
        let rows: Vec<_> = catalog
            .results()
            .iter()
            .filter(|row| row.benchmark == benchmark)
            .collect();
        let resolved = rows.iter().filter(|row| row.model_id.is_some()).count();
        eprintln!("{benchmark:?}: {resolved}/{} rows resolved", rows.len());
        assert!(resolved > 0);
    }
    assert!(
        catalog
            .for_model(Some("anthropic"), "claude-opus-5-5")
            .iter()
            .any(|row| row.benchmark == Benchmark::TerminalBench)
    );
    assert!(
        catalog
            .for_model(Some("openai"), "gpt-6-astra")
            .iter()
            .any(|row| row.benchmark == Benchmark::DeepSwe)
    );
    assert!(
        catalog
            .for_model(Some("openai"), "gpt-5.5")
            .iter()
            .any(|row| row.benchmark == Benchmark::DeepSwe)
    );
    assert!(
        catalog
            .for_model(Some("openrouter"), "mistralai/mistral-large-4-0")
            .iter()
            .any(|row| row.benchmark == Benchmark::TerminalBench)
    );
}
