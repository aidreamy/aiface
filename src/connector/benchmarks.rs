//! Offline lookup of published Terminal-Bench and DeepSWE measurements.

mod identity;
mod sources;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use reqwest::RequestBuilder;

use super::{common::shared_client, discovery::resolve_cache_dir};
use identity::{ModelCatalog, route_id};
use sources::{DEEP_SWE_URL, TERMINAL_READ_URL, parse_deep_swe, parse_terminal};

const MODELS_CATALOG_URL: &str = "https://models.dev/catalog.json";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const CACHE_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
const FEED_LIMIT: usize = 4 * 1024 * 1024;
const CATALOG_LIMIT: usize = 24 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Benchmark {
    TerminalBench,
    DeepSwe,
}

impl Benchmark {
    pub const fn version(self) -> &'static str {
        match self {
            Self::TerminalBench => "4.1",
            Self::DeepSwe => "1.1",
        }
    }
}

/// How the current snapshot was obtained. `fetched_at` is the time of
/// retrieval. The published date of a measurement is independent of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkStatus {
    Fetched { fetched_at: SystemTime },
    Cached { fetched_at: SystemTime, stale: bool },
    Unavailable,
}

/// Bounds of the published 95% confidence interval, expressed as fractions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkInterval {
    pub lower: f64,
    pub upper: f64,
}

/// One published measurement of a model in a specific agent and configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelBenchmark {
    pub benchmark: Benchmark,
    /// A resolved model ID in `publisher/model` form, when available.
    pub model_id: Option<String>,
    pub reported_model: String,
    /// Exact gateway ID when the leaderboard links to a model on that gateway.
    pub reported_model_route: Option<String>,
    /// Agent or harness that actually ran the benchmark.
    pub harness: String,
    pub reasoning_effort: Option<String>,
    /// Terminal-Bench resolution rate or DeepSWE pass@1, in `0.0..=1.0`.
    pub success_rate: f64,
    pub interval: Option<BenchmarkInterval>,
    pub attempts: Option<u64>,
    /// Row update time or published snapshot date supplied by the feed.
    pub published_at: Option<String>,
    pub source_url: String,
}

/// An in-memory snapshot shared across model lookups. Feed failures affect
/// their own status, so an available snapshot remains usable independently.
#[derive(Debug)]
pub struct BenchmarkCatalog {
    models: ModelCatalog,
    results: Vec<ModelBenchmark>,
    identities: BenchmarkStatus,
    terminal: BenchmarkStatus,
    deep_swe: BenchmarkStatus,
}

impl BenchmarkCatalog {
    /// Load both benchmarks and the Models.dev identity catalog concurrently.
    ///
    /// `cache_dir` is relative to the user's local data directory, as in
    /// [`super::refresh_pricing_catalog`]. `None` fetches without a disk cache.
    /// Disk snapshots less than 24 hours old are reused. If a refresh fails,
    /// a previously validated snapshot remains available as stale data.
    pub async fn load(cache_dir: Option<&str>) -> Self {
        Self::load_from(cache_dir.and_then(resolve_cache_dir), true).await
    }

    /// Read the local snapshots for both sources without any network access.
    pub async fn load_cached(cache_dir: &str) -> Self {
        Self::load_from(resolve_cache_dir(cache_dir), false).await
    }

    pub fn status(&self, benchmark: Benchmark) -> BenchmarkStatus {
        match benchmark {
            Benchmark::TerminalBench => self.terminal,
            Benchmark::DeepSwe => self.deep_swe,
        }
    }

    /// Availability of the Models.dev snapshot used to resolve identities.
    pub fn identity_status(&self) -> BenchmarkStatus {
        self.identities
    }

    /// All published measurements, including rows whose model ID is unresolved.
    pub fn results(&self) -> &[ModelBenchmark] {
        &self.results
    }

    /// Return measurements for the resolved model, preserving each evaluated
    /// agent and reasoning effort. A canonical `publisher/model` ID can be
    /// passed with `provider: None`. Pass the active provider for a gateway's
    /// model ID. An unqualified ID without a provider must resolve uniquely.
    pub fn for_model(&self, provider: Option<&str>, model: &str) -> Vec<&ModelBenchmark> {
        let id = self.models.resolve(provider, model);
        let route = route_id(provider, model);
        self.results
            .iter()
            .filter(|result| {
                id.as_ref()
                    .is_some_and(|id| result.model_id.as_ref() == Some(id))
                    || route
                        .as_ref()
                        .is_some_and(|route| result.reported_model_route.as_ref() == Some(route))
            })
            .collect()
    }

    async fn load_from(cache_dir: Option<PathBuf>, online: bool) -> Self {
        let client = shared_client();
        let disk = |file| cache_dir.as_ref().map(|dir| dir.join(file));
        let (models, terminal, deep_swe) = tokio::join!(
            load_document(
                disk("benchmark-models.json"),
                online.then(|| client.get(MODELS_CATALOG_URL)),
                CATALOG_LIMIT,
                ModelCatalog::parse,
            ),
            load_document(
                disk("terminal-bench-4.1.json"),
                online.then(|| {
                    client.post(TERMINAL_READ_URL).json(&serde_json::json!({
                        "package": "terminal-bench/terminal-bench",
                        "name": "4-1-0",
                    }))
                }),
                FEED_LIMIT,
                parse_terminal,
            ),
            load_document(
                disk("deep-swe-1.1.json"),
                online.then(|| client.get(DEEP_SWE_URL)),
                FEED_LIMIT,
                parse_deep_swe,
            )
        );
        let identities = models.status;
        let models = models.data.unwrap_or_default();
        let results = terminal
            .data
            .into_iter()
            .chain(deep_swe.data)
            .flatten()
            .map(|row| row.resolve(&models))
            .collect();
        Self {
            models,
            results,
            identities,
            terminal: terminal.status,
            deep_swe: deep_swe.status,
        }
    }
}

struct Document<T> {
    data: Option<T>,
    status: BenchmarkStatus,
}

async fn load_document<T: Send + 'static>(
    path: Option<PathBuf>,
    request: Option<RequestBuilder>,
    max_bytes: usize,
    parse: fn(&[u8]) -> Option<T>,
) -> Document<T> {
    let unavailable = Document {
        data: None,
        status: BenchmarkStatus::Unavailable,
    };
    let cached = read_cached(path.clone(), max_bytes, parse).await;
    if let Some(request) = request
        && !matches!(
            cached.as_ref().map(|doc| doc.status),
            Some(BenchmarkStatus::Cached { stale: false, .. })
        )
    {
        match download(request, max_bytes).await {
            Ok(bytes) => {
                let cache_path = path.clone();
                if let Ok(Some(document)) = tokio::task::spawn_blocking(move || {
                    let data = parse(&bytes)?;
                    let fetched_at = SystemTime::now();
                    if let Some(path) = cache_path
                        && let Err(error) = write_snapshot(&path, &bytes)
                    {
                        log::warn!("write benchmark cache {}: {error}", path.display());
                    }
                    Some(Document {
                        data: Some(data),
                        status: BenchmarkStatus::Fetched { fetched_at },
                    })
                })
                .await
                {
                    return document;
                }
                log::warn!("benchmark feed returned an invalid snapshot");
            }
            Err(error) => log::warn!("download benchmark snapshot: {error}"),
        }
        // A concurrent refresh may have installed a newer snapshot while
        // this request was in flight.
        return read_cached(path, max_bytes, parse)
            .await
            .or(cached)
            .unwrap_or(unavailable);
    }
    cached.unwrap_or(unavailable)
}

async fn read_cached<T: Send + 'static>(
    path: Option<PathBuf>,
    max_bytes: usize,
    parse: fn(&[u8]) -> Option<T>,
) -> Option<Document<T>> {
    let path = path?;
    tokio::task::spawn_blocking(move || read_snapshot(&path, max_bytes, parse))
        .await
        .ok()
        .flatten()
}

fn read_snapshot<T>(
    path: &Path,
    max_bytes: usize,
    parse: fn(&[u8]) -> Option<T>,
) -> Option<Document<T>> {
    use std::io::Read as _;
    if !path.metadata().ok()?.is_file() {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > max_bytes as u64 {
        return None;
    }
    let fetched_at = metadata.modified().ok()?;
    let stale = fetched_at
        .elapsed()
        .map_or(true, |age| age >= CACHE_MAX_AGE);
    let mut bytes = Vec::new();
    file.take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > max_bytes {
        return None;
    }
    Some(Document {
        data: Some(parse(&bytes)?),
        status: BenchmarkStatus::Cached { fetched_at, stale },
    })
}

fn write_snapshot(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().expect("a snapshot has a cache directory");
    std::fs::create_dir_all(dir)?;
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    temp.write_all(bytes)?;
    temp.persist(path)?;
    Ok(())
}

async fn download(request: RequestBuilder, max_bytes: usize) -> Result<Vec<u8>, DownloadError> {
    let mut response = request
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await?
        .error_for_status()?;
    if response
        .content_length()
        .is_some_and(|len| len > max_bytes as u64)
    {
        return Err(DownloadError::TooLarge);
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if chunk.len() > max_bytes.saturating_sub(bytes.len()) {
            return Err(DownloadError::TooLarge);
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[derive(Debug, thiserror::Error)]
enum DownloadError {
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error("response exceeded the catalog size limit")]
    TooLarge,
}

#[cfg(test)]
mod tests;
