use std::collections::BTreeMap;

use super::{
  BackendRequestLayer, CapabilityAttachmentContract, ModelRegistryRouteContract, ModelRegistryVariantContract,
  capability_with_attachments, variant,
};

pub(super) fn openrouter_variants(
  image_attachment: &CapabilityAttachmentContract,
) -> Vec<ModelRegistryVariantContract> {
  [
    ("gpt-6-luna", "openai/gpt-6-luna", &[][..], true, false),
    ("gpt-6.1-terra", "openai/gpt-6.1-terra", &[][..], true, false),
    (
      "claude-sonnet-4.6",
      "anthropic/claude-sonnet-4.6",
      &["claude-sonnet-4-6"][..],
      true,
      false,
    ),
    ("gpt-4.1", "openai/gpt-4.1", &[][..], false, true),
    ("gpt-4.1-mini", "openai/gpt-4.1-mini", &[][..], false, true),
    ("gpt-4o", "openai/gpt-4o", &[][..], false, false),
    ("gpt-4o-mini", "openai/gpt-4o-mini", &[][..], false, true),
  ]
  .into_iter()
  .map(|(logical, wire, aliases, reasoning, rerank)| {
    let mut outputs = vec!["text", "object", "structured"];
    if rerank {
      outputs.push("rerank");
    }
    let flags = if reasoning {
      &["tool_calling_supported", "reasoning_supported"][..]
    } else {
      &["tool_calling_supported"][..]
    };
    let mut model = variant(
      "openrouter",
      logical,
      wire,
      aliases,
      &[],
      vec![capability_with_attachments(
        &["text", "image"],
        &outputs,
        image_attachment.clone(),
        Some(image_attachment.clone()),
        false,
      )],
      None,
      flags,
      None,
    );
    model.route_overrides = Some(
      ["text", "structured", "rerank"]
        .into_iter()
        .map(|operation| {
          (
            operation.to_string(),
            ModelRegistryRouteContract {
              request_layers: vec![
                BackendRequestLayer::OpenrouterResponses,
                BackendRequestLayer::OpenrouterChatCompletions,
                BackendRequestLayer::OpenrouterAnthropic,
              ],
            },
          )
        })
        .collect::<BTreeMap<_, _>>(),
    );
    model
  })
  .collect()
}
