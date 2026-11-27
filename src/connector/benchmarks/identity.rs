use std::collections::HashMap;

use serde::Deserialize;

/// The Models.dev catalog joins provider-specific IDs and published model
/// identities. The scoring feeds carry their own run configurations.
#[derive(Debug, Default, Deserialize)]
pub(super) struct ModelCatalog {
    providers: HashMap<String, Provider>,
    models: HashMap<String, Model>,
}

#[derive(Debug, Deserialize)]
struct Provider {
    #[serde(default)]
    models: HashMap<String, ProviderModel>,
}

#[derive(Debug, Deserialize)]
struct ProviderModel {
    canonical_model_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Model {
    name: String,
}

impl ModelCatalog {
    pub(super) fn parse(bytes: &[u8]) -> Option<Self> {
        let catalog: Self = serde_json::from_slice(bytes).ok()?;
        (!catalog.models.is_empty() && !catalog.providers.is_empty()).then_some(catalog)
    }

    pub(super) fn resolve(&self, provider: Option<&str>, model: &str) -> Option<String> {
        let model = model.trim();
        if model.is_empty() {
            return None;
        }
        let Some(provider) = provider else {
            if model.contains('/') {
                return Some(model.to_ascii_lowercase());
            }
            return self.unique_id(|id, _| bare_id(id).eq_ignore_ascii_case(model));
        };

        let provider = provider.trim().to_ascii_lowercase();
        let provider = match provider.as_str() {
            "claude" => "anthropic",
            "copilot" => "github-copilot",
            "gemini" => "google",
            "kimi" => "moonshotai",
            "zen" => "opencode",
            other => other,
        };
        let provider_model = model_without_provider(provider, model);
        let canonical = self.providers.get(provider).and_then(|catalog| {
            catalog
                .models
                .get(model)
                .or_else(|| catalog.models.get(provider_model))?
                .canonical_model_id
                .as_deref()
        });
        if let Some(id) = canonical {
            // Canonical IDs can collapse provider variants such as a Pro or
            // dated deployment. Match only aliases that preserve the slug.
            return (name_key(bare_id(id)) == name_key(bare_id(provider_model)))
                .then(|| id.to_ascii_lowercase());
        }
        Some(format!("{provider}/{provider_model}").to_ascii_lowercase())
    }

    pub(super) fn deep_swe_model(&self, provider: Option<&str>, model: &str) -> Option<String> {
        if let Some(provider) = provider {
            return self.resolve(Some(provider), model);
        }
        if model.contains('/') {
            return self.resolve(None, model);
        }
        let slug = name_key(model);
        self.unique_id(|id, _| name_key(bare_id(id)) == slug)
    }

    pub(super) fn terminal_model(&self, label: &str, organization: &str) -> Option<String> {
        let name = name_key(label);
        let publisher = match organization.trim().to_ascii_lowercase().as_str() {
            "anthropic" => Some("anthropic"),
            "google" | "google deepmind" => Some("google"),
            "openai" => Some("openai"),
            _ => None,
        };
        self.unique_id(|id, model| {
            if publisher
                .is_some_and(|publisher| id.split_once('/').is_none_or(|(p, _)| p != publisher))
            {
                return false;
            }
            let catalog_name = name_key(&model.name);
            catalog_name == name
                || name_key(bare_id(id)) == name
                || (publisher == Some("anthropic")
                    && catalog_name.strip_prefix("claude-") == Some(name.as_str()))
        })
    }

    fn unique_id(&self, matches: impl Fn(&str, &Model) -> bool) -> Option<String> {
        let mut ids = self.models.iter().filter(|(id, model)| matches(id, model));
        let (id, _) = ids.next()?;
        ids.next().is_none().then(|| id.to_ascii_lowercase())
    }
}

pub(super) fn route_id(provider: Option<&str>, model: &str) -> Option<String> {
    let model = model.trim();
    if let Some(provider) = provider {
        let provider = provider.trim().to_ascii_lowercase();
        return Some(format!(
            "{provider}/{}",
            model_without_provider(&provider, model)
        ));
    }
    model.contains('/').then(|| model.to_owned())
}

fn model_without_provider<'a>(provider: &str, model: &'a str) -> &'a str {
    match model.split_once('/') {
        Some((prefix, id)) if prefix.eq_ignore_ascii_case(provider) => id,
        _ => model,
    }
}

fn bare_id(id: &str) -> &str {
    id.rsplit('/').next().unwrap_or(id)
}

/// Preserve the complete version and variant while reconciling punctuation in
/// public names such as `Gemini 3.8 Flash` and `gemini-3-8-flash`.
fn name_key(name: &str) -> String {
    name.to_ascii_lowercase()
        .split([' ', '.', '-', '_'])
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}
