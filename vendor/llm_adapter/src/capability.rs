use std::collections::HashSet;

#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInput {
  Text,
  Image,
  Audio,
  File,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOutput {
  Text,
  Object,
  Structured,
  Embedding,
  Rerank,
  Image,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFeature {
  ToolCalling,
  Reasoning,
  WebSearch,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentKind {
  Image,
  Audio,
  File,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentSource {
  Url,
  Data,
  Bytes,
  FileHandle,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeclaredModelCapability {
  pub input: Vec<ModelInput>,
  pub output: Vec<ModelOutput>,
  pub features: Vec<ModelFeature>,
  pub attachment_kinds: Vec<AttachmentKind>,
  pub attachment_sources: Vec<AttachmentSource>,
}

#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelRequirements {
  pub input: Vec<ModelInput>,
  pub output: Vec<ModelOutput>,
  pub features: Vec<ModelFeature>,
  pub attachment_kinds: Vec<AttachmentKind>,
  pub attachment_sources: Vec<AttachmentSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityValidationError {
  #[error("{field} must not be empty")]
  Empty { field: &'static str },
  #[error("{field} contains a duplicate value")]
  Duplicate { field: &'static str },
  #[error("attachment sources require at least one attachment kind")]
  SourceWithoutKind,
  #[error("attachment kinds require at least one attachment source")]
  KindWithoutSource,
  #[error("attachment kind {kind:?} requires model input {input:?}")]
  AttachmentInputMissing { kind: AttachmentKind, input: ModelInput },
  #[error("declared capability exceeds the allowed upper bound")]
  ExceedsUpperBound,
}

fn ensure_unique<T: Eq + std::hash::Hash>(values: &[T], field: &'static str) -> Result<(), CapabilityValidationError> {
  let mut seen = HashSet::with_capacity(values.len());
  if values.iter().all(|value| seen.insert(value)) {
    Ok(())
  } else {
    Err(CapabilityValidationError::Duplicate { field })
  }
}

pub fn validate_declared_capability(capability: &DeclaredModelCapability) -> Result<(), CapabilityValidationError> {
  if capability.input.is_empty() {
    return Err(CapabilityValidationError::Empty { field: "input" });
  }
  if capability.output.is_empty() {
    return Err(CapabilityValidationError::Empty { field: "output" });
  }
  ensure_unique(&capability.input, "input")?;
  ensure_unique(&capability.output, "output")?;
  ensure_unique(&capability.features, "features")?;
  ensure_unique(&capability.attachment_kinds, "attachmentKinds")?;
  ensure_unique(&capability.attachment_sources, "attachmentSources")?;

  if capability.attachment_kinds.is_empty() && !capability.attachment_sources.is_empty() {
    return Err(CapabilityValidationError::SourceWithoutKind);
  }
  if !capability.attachment_kinds.is_empty() && capability.attachment_sources.is_empty() {
    return Err(CapabilityValidationError::KindWithoutSource);
  }
  for kind in &capability.attachment_kinds {
    let input = match kind {
      AttachmentKind::Image => ModelInput::Image,
      AttachmentKind::Audio => ModelInput::Audio,
      AttachmentKind::File => ModelInput::File,
    };
    if !capability.input.contains(&input) {
      return Err(CapabilityValidationError::AttachmentInputMissing { kind: *kind, input });
    }
  }
  Ok(())
}

pub fn capability_matches(capability: &DeclaredModelCapability, requirements: &ModelRequirements) -> bool {
  requirements.input.iter().all(|value| capability.input.contains(value))
    && requirements
      .output
      .iter()
      .all(|value| capability.output.contains(value))
    && requirements
      .features
      .iter()
      .all(|value| capability.features.contains(value))
    && requirements
      .attachment_kinds
      .iter()
      .all(|value| capability.attachment_kinds.contains(value))
    && requirements
      .attachment_sources
      .iter()
      .all(|value| capability.attachment_sources.contains(value))
}

pub fn declared_model_matches(capabilities: &[DeclaredModelCapability], requirements: &ModelRequirements) -> bool {
  capabilities
    .iter()
    .any(|capability| capability_matches(capability, requirements))
}

pub fn validate_capability_upper_bound(
  capability: &DeclaredModelCapability,
  upper_bound: &[DeclaredModelCapability],
) -> Result<(), CapabilityValidationError> {
  validate_declared_capability(capability)?;
  let requirements = ModelRequirements {
    input: capability.input.clone(),
    output: capability.output.clone(),
    features: capability.features.clone(),
    attachment_kinds: capability.attachment_kinds.clone(),
    attachment_sources: capability.attachment_sources.clone(),
  };
  if declared_model_matches(upper_bound, &requirements) {
    Ok(())
  } else {
    Err(CapabilityValidationError::ExceedsUpperBound)
  }
}

pub fn provider_default_capability_upper_bound(provider: &str, model_id: &str) -> Option<Vec<DeclaredModelCapability>> {
  use crate::target::BackendOperation;
  let provider = provider.parse::<crate::backend::BackendProvider>().ok()?;
  crate::core::default_model_registry_variants()
    .into_iter()
    .filter(|variant| provider.matches_registry_backend(&variant.backend_kind) && variant.raw_model_id == model_id)
    .find_map(|variant| {
      let mut allowed_outputs = Vec::new();
      for operation in [
        BackendOperation::Chat,
        BackendOperation::Structured,
        BackendOperation::Embedding,
        BackendOperation::Rerank,
        BackendOperation::Image,
      ] {
        if let Ok(layer) = provider.default_request_layer(operation, None)
          && let Some(route) = variant
            .operation_routes(operation)
            .ok()?
            .into_iter()
            .find(|route| route.request_layer == layer && route.wire_model == model_id)
        {
          allowed_outputs.extend(route.capabilities.into_iter().flat_map(|capability| capability.output));
        }
      }
      let capabilities = model_capability_upper_bound(&variant);
      capabilities
        .iter()
        .all(|capability| capability.output.iter().all(|output| allowed_outputs.contains(output)))
        .then_some(capabilities)
    })
}

pub fn model_capability_upper_bound(variant: &crate::core::ModelRegistryVariant) -> Vec<DeclaredModelCapability> {
  model_capabilities_upper_bound(
    &variant.capabilities,
    &variant.backend_kind,
    variant.behavior_flags.as_deref().unwrap_or_default(),
  )
}

pub(crate) fn model_capabilities_upper_bound(
  capabilities: &[crate::core::ModelCapability],
  backend_kind: &str,
  behavior_flags: &[String],
) -> Vec<DeclaredModelCapability> {
  capabilities
    .iter()
    .map(|capability| {
      let input = capability
        .input
        .iter()
        .filter_map(|value| match value.as_str() {
          "text" => Some(ModelInput::Text),
          "image" => Some(ModelInput::Image),
          "audio" => Some(ModelInput::Audio),
          "file" => Some(ModelInput::File),
          _ => None,
        })
        .collect::<Vec<_>>();
      let output = capability
        .output
        .iter()
        .filter_map(|value| match value.as_str() {
          "text" => Some(ModelOutput::Text),
          "object" => Some(ModelOutput::Object),
          "structured" => Some(ModelOutput::Structured),
          "embedding" => Some(ModelOutput::Embedding),
          "rerank" => Some(ModelOutput::Rerank),
          "image" => Some(ModelOutput::Image),
          _ => None,
        })
        .collect::<Vec<_>>();
      let mut features = Vec::new();
      if (matches!(
        backend_kind,
        "openai_responses"
          | "openai_chat"
          | "anthropic"
          | "anthropic_vertex"
          | "gemini_api"
          | "gemini_vertex"
          | "cloudflare_workers_ai"
      ) || behavior_flags.iter().any(|flag| flag == "tool_calling_supported"))
        && output
          .iter()
          .any(|value| matches!(value, ModelOutput::Text | ModelOutput::Object | ModelOutput::Structured))
      {
        features.push(ModelFeature::ToolCalling);
      }
      if behavior_flags.iter().any(|flag| flag == "reasoning_supported")
        && output
          .iter()
          .any(|value| matches!(value, ModelOutput::Text | ModelOutput::Object | ModelOutput::Structured))
      {
        features.push(ModelFeature::Reasoning);
      }
      let attachment = capability
        .attachments
        .clone()
        .or_else(|| capability.structured_attachments.clone());
      let attachment_kinds = attachment
        .as_ref()
        .map(|attachment| {
          attachment
            .kinds
            .iter()
            .filter_map(|value| match value.as_str() {
              "image" => Some(AttachmentKind::Image),
              "audio" => Some(AttachmentKind::Audio),
              "file" => Some(AttachmentKind::File),
              _ => None,
            })
            .collect()
        })
        .unwrap_or_default();
      let attachment_sources = attachment
        .and_then(|attachment| attachment.source_kinds)
        .unwrap_or_default()
        .iter()
        .filter_map(|value| match value.as_str() {
          "url" => Some(AttachmentSource::Url),
          "data" => Some(AttachmentSource::Data),
          "bytes" => Some(AttachmentSource::Bytes),
          "file_handle" => Some(AttachmentSource::FileHandle),
          _ => None,
        })
        .collect();
      DeclaredModelCapability {
        input,
        output,
        features,
        attachment_kinds,
        attachment_sources,
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn capability(input: Vec<ModelInput>, output: Vec<ModelOutput>) -> DeclaredModelCapability {
    DeclaredModelCapability {
      input,
      output,
      features: Vec::new(),
      attachment_kinds: Vec::new(),
      attachment_sources: Vec::new(),
    }
  }

  #[test]
  fn does_not_union_capability_entries() {
    let declared = vec![
      capability(vec![ModelInput::Text], vec![ModelOutput::Text]),
      capability(vec![ModelInput::Image], vec![ModelOutput::Image]),
    ];
    let requirements = ModelRequirements {
      input: vec![ModelInput::Text],
      output: vec![ModelOutput::Image],
      features: Vec::new(),
      attachment_kinds: Vec::new(),
      attachment_sources: Vec::new(),
    };
    assert!(!declared_model_matches(&declared, &requirements));
  }

  #[test]
  fn validates_attachment_consistency_and_duplicates() {
    let cases = [
      DeclaredModelCapability {
        input: vec![ModelInput::Text],
        output: vec![ModelOutput::Text],
        features: Vec::new(),
        attachment_kinds: vec![AttachmentKind::Image],
        attachment_sources: Vec::new(),
      },
      DeclaredModelCapability {
        input: vec![ModelInput::Text, ModelInput::Text],
        output: vec![ModelOutput::Text],
        features: Vec::new(),
        attachment_kinds: Vec::new(),
        attachment_sources: Vec::new(),
      },
    ];
    assert!(cases.iter().all(|case| validate_declared_capability(case).is_err()));
  }

  #[test]
  fn projects_provider_default_registry_as_declared_upper_bound() {
    assert!(crate::backend::BackendProvider::OpenAi.matches_registry_backend("openai_responses"));
    let upper = provider_default_capability_upper_bound("openai", "gpt-5-mini").unwrap();
    assert!(
      upper
        .iter()
        .any(|capability| capability.features.contains(&ModelFeature::ToolCalling))
    );
    let multi = provider_default_capability_upper_bound("openai", "gpt-4.1").unwrap();
    assert!(multi.iter().any(|capability| {
      [
        ModelOutput::Text,
        ModelOutput::Object,
        ModelOutput::Structured,
        ModelOutput::Rerank,
      ]
      .iter()
      .all(|output| capability.output.contains(output))
    }));
    for model in ["o1", "o3", "o4-mini"] {
      assert!(provider_default_capability_upper_bound("openai", model).is_none());
    }
    assert!(provider_default_capability_upper_bound("openai", "unknown-model").is_none());

    for (provider, model) in [
      ("fal", "lora/image-to-image"),
      ("openai", "gpt-image-1"),
      ("openai", "gpt-image-2"),
      ("openai", "gpt-image-2.6-flare"),
    ] {
      let upper = provider_default_capability_upper_bound(provider, model).unwrap();
      for source in [AttachmentSource::Data, AttachmentSource::Bytes] {
        assert!(
          declared_model_matches(
            &upper,
            &ModelRequirements {
              input: vec![ModelInput::Text, ModelInput::Image],
              output: vec![ModelOutput::Image],
              features: Vec::new(),
              attachment_kinds: vec![AttachmentKind::Image],
              attachment_sources: vec![source],
            }
          ),
          "{provider}/{model} must accept host-materialized image input"
        );
      }
    }
  }
}
