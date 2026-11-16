mod client;
mod dispatch;
mod fal_upload;
mod gemini_files;
mod provider_config;
mod provider_types;
mod providers;
mod request_layer;
pub use provider_config::{backend_provider_config_schema, backend_providers_config_schema, validate_backend_config};
pub use provider_types::{ApiShape, BackendProvider, CredentialPlan};

use crate::{
  router::ExecutableProtocol,
  target::{BackendOperation, OpenAiDialect, TargetCompileError, canonicalize_endpoint},
};
mod types;

pub use client::DefaultHttpClient;
#[cfg(feature = "reqwest-client")]
pub use client::ReqwestHttpClient;
#[cfg(feature = "ureq-client")]
pub use client::UreqHttpClient;
pub use dispatch::{
  collect_stream_events, dispatch_embedding_request, dispatch_image_request, dispatch_request, dispatch_rerank_request,
  dispatch_stream_events_with, dispatch_structured_request,
};
pub use request_layer::{
  AttachmentReferenceMode, AttachmentReferencePlan, AttachmentReferenceReason, RequestIntent, RequestIntentReasoning,
  ResolvedRequestIntent, resolve_attachment_reference_plan, resolve_request_intent,
};
pub use types::{
  BackendConfig, BackendError, BackendHttpClient, BackendRequestLayer, ChatProtocol, EmbeddingProtocol, HttpBody,
  HttpFailureContext, HttpFailureKind, HttpFailureStage, HttpMethod, HttpRawRequest, HttpRawResponse, HttpRequest,
  HttpResponse, HttpStreamResponse, HttpUploadRequest, ImageProtocol, MultipartPart, RequestTransmission,
  RerankProtocol, SensitiveString, StructuredProtocol,
};

#[cfg(test)]
mod tests;
