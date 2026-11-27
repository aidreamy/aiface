use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{Benchmark, BenchmarkInterval, ModelBenchmark, identity::ModelCatalog};

// The same public Harbor Hub endpoint used by the official Terminal-Bench site.
pub(super) const TERMINAL_READ_URL: &str =
    "https://ofhuhcpkvzjlejydnvyd.supabase.co/functions/v1/leaderboard-read";
const TERMINAL_BOARD_URL: &str = "https://www.tbench.ai/?version=4.1";
const TERMINAL_ROW_URL: &str = "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/5/leaderboards/4-1-0/rows/";
pub(super) const DEEP_SWE_URL: &str =
    "https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json";

pub(super) struct Measurement {
    model_publisher: Option<String>,
    result: ModelBenchmark,
}

impl Measurement {
    pub(super) fn resolve(mut self, models: &ModelCatalog) -> ModelBenchmark {
        self.result.model_id = match self.result.benchmark {
            Benchmark::TerminalBench => models.terminal_model(
                &self.result.reported_model,
                self.model_publisher.as_deref().unwrap_or_default(),
            ),
            Benchmark::DeepSwe => {
                models.deep_swe_model(self.model_publisher.as_deref(), &self.result.reported_model)
            }
        };
        self.result
    }
}

#[derive(Deserialize)]
struct TerminalBoard {
    leaderboard: LeaderboardId,
    rows: Vec<Value>,
    pagination: Option<Pagination>,
}

#[derive(Deserialize)]
struct LeaderboardId {
    name: String,
    package: String,
}

#[derive(Deserialize)]
struct Pagination {
    total_pages: usize,
}

#[derive(Deserialize)]
struct TerminalRow {
    id: String,
    status: String,
    metadata: TerminalMetadata,
    metrics: TerminalMetrics,
    updated_at: Option<String>,
    n_trials: Option<u64>,
}

#[derive(Deserialize)]
struct TerminalMetadata {
    model_display: DisplayLabel,
    model_org: Option<DisplayLabel>,
    agent_display: DisplayLabel,
    reasoning_effort: Option<String>,
}

#[derive(Deserialize)]
struct DisplayLabel {
    label: String,
    url: Option<String>,
}

#[derive(Deserialize)]
struct TerminalMetrics {
    accuracy: f64,
    accuracy_ci95_half_width: Option<f64>,
}

pub(super) fn parse_terminal(bytes: &[u8]) -> Option<Vec<Measurement>> {
    let board: TerminalBoard = serde_json::from_slice(bytes).ok()?;
    if board.leaderboard.name != "4-1-0"
        || board.leaderboard.package != "terminal-bench/terminal-bench"
        || board.pagination.is_some_and(|page| page.total_pages > 1)
    {
        return None;
    }
    let visible = board
        .rows
        .into_iter()
        .filter(|row| row.get("status").and_then(Value::as_str) != Some("hide"));
    measurements(visible, |row: TerminalRow| {
        if row.status != "display" {
            return None;
        }
        let success_rate = rate(row.metrics.accuracy / 100.0)?;
        let interval = row
            .metrics
            .accuracy_ci95_half_width
            .and_then(|width| rate(width / 100.0))
            .and_then(|width| {
                interval(
                    (success_rate - width).max(0.0),
                    (success_rate + width).min(1.0),
                    success_rate,
                )
            });
        let source_url = if !row.id.is_empty()
            && row
                .id
                .bytes()
                .all(|ch| ch.is_ascii_hexdigit() || ch == b'-')
        {
            format!("{TERMINAL_ROW_URL}{}", row.id)
        } else {
            TERMINAL_BOARD_URL.into()
        };
        Some(Measurement {
            model_publisher: row.metadata.model_org.and_then(|org| nonempty(&org.label)),
            result: ModelBenchmark {
                benchmark: Benchmark::TerminalBench,
                model_id: None,
                reported_model: nonempty(&row.metadata.model_display.label)?,
                reported_model_route: row
                    .metadata
                    .model_display
                    .url
                    .as_deref()
                    .and_then(openrouter_model_route),
                harness: nonempty(&row.metadata.agent_display.label)?,
                reasoning_effort: row.metadata.reasoning_effort.as_deref().and_then(nonempty),
                success_rate,
                interval,
                attempts: row.n_trials,
                published_at: row.updated_at,
                source_url,
            },
        })
    })
}

#[derive(Deserialize)]
struct DeepSweBoard {
    generated_at: String,
    rows: Vec<Value>,
}

#[derive(Deserialize)]
struct DeepSweRow {
    model: String,
    harness: String,
    provider: Option<String>,
    reasoning_effort: Option<String>,
    pass_at_1: f64,
    ci_lo: Option<f64>,
    ci_hi: Option<f64>,
    n_attempted: Option<u64>,
}

pub(super) fn parse_deep_swe(bytes: &[u8]) -> Option<Vec<Measurement>> {
    let board: DeepSweBoard = serde_json::from_slice(bytes).ok()?;
    let published_at = nonempty(&board.generated_at)?;
    measurements(board.rows, |row: DeepSweRow| {
        let success_rate = rate(row.pass_at_1)?;
        let interval = row
            .ci_lo
            .zip(row.ci_hi)
            .and_then(|(lower, upper)| interval(lower, upper, success_rate));
        Some(Measurement {
            model_publisher: row.provider.as_deref().and_then(nonempty),
            result: ModelBenchmark {
                benchmark: Benchmark::DeepSwe,
                model_id: None,
                reported_model: nonempty(&row.model)?,
                reported_model_route: None,
                harness: nonempty(&row.harness)?,
                reasoning_effort: row.reasoning_effort.as_deref().and_then(nonempty),
                success_rate,
                interval,
                attempts: row.n_attempted,
                published_at: Some(published_at.clone()),
                source_url: DEEP_SWE_URL.into(),
            },
        })
    })
}

fn measurements<T: DeserializeOwned>(
    raw: impl IntoIterator<Item = Value>,
    parse: impl Fn(T) -> Option<Measurement>,
) -> Option<Vec<Measurement>> {
    let mut raw = raw.into_iter().peekable();
    if raw.peek().is_none() {
        return Some(Vec::new());
    }
    let reports: Vec<_> = raw
        .filter_map(|row| parse(serde_json::from_value(row).ok()?))
        .collect();
    (!reports.is_empty()).then_some(reports)
}

fn nonempty(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn openrouter_model_route(model_url: &str) -> Option<String> {
    let url = url::Url::parse(model_url).ok()?;
    if url.scheme() != "https" || url.host_str() != Some("openrouter.ai") {
        return None;
    }
    let model = url.path().strip_prefix('/')?;
    let (publisher, slug) = model.split_once('/')?;
    if publisher.is_empty() || slug.is_empty() || slug.contains('/') {
        return None;
    }
    Some(format!("openrouter/{model}"))
}

fn rate(value: f64) -> Option<f64> {
    (0.0..=1.0).contains(&value).then_some(value)
}

fn interval(lower: f64, upper: f64, measured: f64) -> Option<BenchmarkInterval> {
    (rate(lower)? <= measured && measured <= rate(upper)?)
        .then_some(BenchmarkInterval { lower, upper })
}
