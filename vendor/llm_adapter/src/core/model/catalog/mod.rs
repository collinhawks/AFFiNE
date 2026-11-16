mod anthropic;
mod deepseek;
mod gemini;
mod helpers;
mod kimi;
mod openai;
mod openai_image;
mod opencode_go;
mod opencode_zen;
mod openrouter;
mod other;

use helpers::{CapabilityAttachmentContract, ModelRegistryRouteContract, capability_with_attachments, variant};

use self::{
  anthropic::anthropic_variants,
  deepseek::deepseek_variants,
  gemini::gemini_variants,
  helpers::{ModelRegistryVariantContract, attachment},
  kimi::kimi_variants,
  openai::openai_variants,
  openai_image::openai_image_variants,
  opencode_go::opencode_go_variants,
  opencode_zen::opencode_zen_variants,
  openrouter::openrouter_variants,
  other::{cloudflare_variants, fal_variants, morph_variants, perplexity_variants},
};
use super::{BackendRequestLayer, CapabilityAttachment, ModelCapability, ModelRegistryRoute, ModelRegistryVariant};

pub(crate) fn registry_variants() -> Vec<ModelRegistryVariantContract> {
  let image_attachment = attachment(&["image"], &["url", "data", "bytes"], true);
  let gemini_attachment = attachment(
    &["image", "audio", "file"],
    &["url", "data", "bytes", "file_handle"],
    true,
  );

  [
    openai_variants(&image_attachment),
    openai_image_variants(&image_attachment),
    cloudflare_variants(&image_attachment),
    fal_variants(&image_attachment),
    gemini_variants(&gemini_attachment),
    perplexity_variants(),
    anthropic_variants(&image_attachment),
    morph_variants(),
    deepseek_variants(&image_attachment),
    kimi_variants(&image_attachment),
    opencode_go_variants(),
    opencode_zen_variants(),
    openrouter_variants(&image_attachment),
  ]
  .into_iter()
  .flatten()
  .collect()
}
