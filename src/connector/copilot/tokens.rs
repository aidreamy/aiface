//! Token extraction for the GitHub family.
//!
//! The Copilot runtime reports usage on `assistant.usage` events, whose
//! payload is the `AssistantUsageData` schema serialized in camelCase
//! (what the runtime sends over the wire — and what the caller stores as
//! the `raw` payload). The normalized [`TokenUsage`] contract mirrors the
//! other families:
//!
//! - `input_tokens` EXCLUDES cached tokens (`cacheReadTokens` ⊆
//!   `inputTokens`), so `total_input_tokens()` stays correct;
//! - `cache_write_tokens` maps the runtime's `cacheWriteTokens`;
//! - `reasoning_tokens` carries the runtime's reasoning counter;
//! - `reported_cost` converts the runtime's REAL billed amount,
//!   `copilotUsage.totalNanoAiu`, to USD. The sibling top-level `cost`
//!   field is a billing MULTIPLIER (premium-request units), NOT money, so
//!   it is deliberately ignored.

use crate::connector::usage::TokenUsage;

/// 1,000,000,000 nano-AI units make one AI credit (the SDK's
/// `TokenPriceUnit::NanoAiu` contract).
const NANO_AIU_PER_AI_CREDIT: f64 = 1_000_000_000.0;

/// GitHub bills one AI credit at USD 0.01
/// (docs.github.com > Copilot > billing).
const USD_PER_AI_CREDIT: f64 = 0.01;

/// Convert a `totalNanoAiu` amount to USD.
fn nano_aiu_to_usd(nano_aiu: f64) -> f64 {
    nano_aiu / NANO_AIU_PER_AI_CREDIT * USD_PER_AI_CREDIT
}

/// Extract completion tokens from a raw usage payload.
///
/// Returns `None` when no usage object is present or nothing was emitted.
#[must_use]
pub fn extract_tokens(raw: &str) -> Option<u32> {
    extract_usage(raw)
        .map(|u| u.output_tokens)
        .filter(|n| *n > 0)
}

/// Extract the full token usage from a raw usage payload.
#[must_use]
pub fn extract_usage(raw: &str) -> Option<TokenUsage> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CopilotUsage {
        #[serde(default)]
        total_nano_aiu: Option<f64>,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Usage {
        #[serde(default)]
        input_tokens: Option<i64>,
        #[serde(default)]
        output_tokens: Option<i64>,
        #[serde(default)]
        cache_read_tokens: Option<i64>,
        #[serde(default)]
        cache_write_tokens: Option<i64>,
        #[serde(default)]
        reasoning_tokens: Option<i64>,
        #[serde(default)]
        copilot_usage: Option<CopilotUsage>,
    }
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let usage: Usage = serde_json::from_value(v.clone()).ok()?;
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    if input == 0 && output == 0 {
        return None;
    }
    let cache_read = usage.cache_read_tokens.unwrap_or(0);
    Some(TokenUsage {
        input_tokens: input.saturating_sub(cache_read).try_into().unwrap_or(0),
        output_tokens: output.try_into().unwrap_or(0),
        cache_creation_input_tokens: usage
            .cache_write_tokens
            .unwrap_or(0)
            .try_into()
            .unwrap_or(0),
        cache_read_input_tokens: cache_read.try_into().unwrap_or(0),
        reasoning_tokens: usage.reasoning_tokens.unwrap_or(0).try_into().unwrap_or(0),
        reported_cost: usage
            .copilot_usage
            .and_then(|billing| billing.total_nano_aiu)
            .map(nano_aiu_to_usd)
            .filter(|usd| usd.is_finite() && *usd >= 0.0),
        reported_cost_credits: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_usage_with_cache_and_reasoning() {
        // 420,000,000 nano-AIU = 0.42 AI credits = USD 0.0042.
        let json = r#"{"inputTokens":100,"outputTokens":20,"cacheReadTokens":40,
            "cacheWriteTokens":5,"reasoningTokens":8,
            "copilotUsage":{"totalNanoAiu":420000000}}"#;
        let usage = extract_usage(json).expect("usage present");
        assert_eq!(usage.input_tokens, 60);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cache_read_input_tokens, 40);
        assert_eq!(usage.cache_creation_input_tokens, 5);
        assert_eq!(usage.reasoning_tokens, 8);
        assert_eq!(usage.reported_cost, Some(0.0042));
    }

    #[test]
    fn multiplier_cost_is_not_money() {
        // The runtime's top-level `cost` is a billing multiplier (premium
        // requests), not USD. A 15x multiplier must never become $15.
        let json = r#"{"inputTokens":100,"outputTokens":20,"cost":15}"#;
        let usage = extract_usage(json).expect("usage present");
        assert_eq!(usage.reported_cost, None);
    }

    #[test]
    fn zero_nano_aiu_is_a_real_zero_cost() {
        let json = r#"{"inputTokens":10,"outputTokens":5,
            "copilotUsage":{"totalNanoAiu":0}}"#;
        let usage = extract_usage(json).expect("usage present");
        assert_eq!(usage.reported_cost, Some(0.0));
    }

    #[test]
    fn empty_usage_is_none() {
        assert_eq!(extract_usage("{}"), None);
        assert_eq!(extract_usage(r#"{"inputTokens":0,"outputTokens":0}"#), None);
    }

    #[test]
    fn invalid_json_is_none() {
        assert_eq!(extract_tokens("not json"), None);
    }

    #[test]
    fn tokens_filter_zero_output() {
        assert_eq!(
            extract_tokens(r#"{"inputTokens":10,"outputTokens":0}"#),
            None
        );
        assert_eq!(
            extract_tokens(r#"{"inputTokens":10,"outputTokens":3}"#),
            Some(3)
        );
    }
}
