use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
  ApiShape, BackendConfig, BackendError, BackendHttpClient, BackendOperation, BackendRequestLayer, ChatProtocol,
  EmbeddingProtocol, ExecutableProtocol, ImageProtocol, RerankProtocol, StructuredProtocol,
};
use crate::core::{RerankRequest, RerankResponse};

mod anthropic;
mod attachments;
mod bindings;
mod chat_completions;
mod cloudflare_workers_ai;
mod gemini_api;
mod gemini_vertex;
mod openai_embeddings;
mod openrouter;
mod perplexity_sonar;
mod responses;
mod vertex_anthropic;

use self::{
  anthropic::AnthropicRequestLayer,
  attachments::{generic_remote_reference_plan, plan_gemini_api_attachment_reference},
  chat_completions::ChatCompletionsRequestLayer,
  cloudflare_workers_ai::CloudflareWorkersAiRequestLayer,
  gemini_api::GeminiApiRequestLayer,
  gemini_vertex::GeminiVertexRequestLayer,
  openai_embeddings::OpenaiEmbeddingsRequestLayer,
  openrouter::{
    OpenrouterAnthropicRequestLayer, OpenrouterChatCompletionsRequestLayer, OpenrouterResponsesRequestLayer,
  },
  perplexity_sonar::PerplexitySonarRequestLayer,
  responses::ResponsesRequestLayer,
  vertex_anthropic::VertexAnthropicRequestLayer,
};

trait RequestLayerImpl {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String;
  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)>;

  fn rewrite_body(&self, body: Value) -> Value {
    body
  }

  fn build_embedding_url(&self, _base_url: &str, _model: &str) -> Result<String, BackendError> {
    Err(BackendError::InvalidConfig {
      message: "embedding dispatch is not supported for this request layer".to_string(),
    })
  }

  fn build_embedding_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.build_headers(config, false)
  }

  fn rewrite_embedding_body(&self, body: Value) -> Value {
    body
  }

  fn build_rerank_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    Ok(self.build_url(base_url, model, false))
  }

  fn build_rerank_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.build_headers(config, false)
  }

  fn rewrite_rerank_body(&self, body: Value) -> Value {
    self.rewrite_body(body)
  }

  fn dispatch_rerank(
    &self,
    _client: &dyn BackendHttpClient,
    _config: &BackendConfig,
    _protocol: &RerankProtocol,
    _request: &RerankRequest,
  ) -> Result<Option<RerankResponse>, BackendError> {
    Ok(None)
  }
}

const ANTHROPIC_LAYER: AnthropicRequestLayer = AnthropicRequestLayer;
const CHAT_COMPLETIONS_LAYER: ChatCompletionsRequestLayer = ChatCompletionsRequestLayer;
const CLOUDFLARE_WORKERS_AI_LAYER: CloudflareWorkersAiRequestLayer = CloudflareWorkersAiRequestLayer;
const GEMINI_API_LAYER: GeminiApiRequestLayer = GeminiApiRequestLayer;
const GEMINI_VERTEX_LAYER: GeminiVertexRequestLayer = GeminiVertexRequestLayer;
const PERPLEXITY_SONAR_LAYER: PerplexitySonarRequestLayer = PerplexitySonarRequestLayer;
const RESPONSES_LAYER: ResponsesRequestLayer = ResponsesRequestLayer;
const VERTEX_ANTHROPIC_LAYER: VertexAnthropicRequestLayer = VertexAnthropicRequestLayer;
const OPENAI_IMAGES_LAYER: OpenaiImagesRequestLayer = OpenaiImagesRequestLayer;
const FAL_LAYER: FalRequestLayer = FalRequestLayer;
const OPENAI_EMBEDDINGS_LAYER: OpenaiEmbeddingsRequestLayer = OpenaiEmbeddingsRequestLayer;
const OPENROUTER_ANTHROPIC_LAYER: OpenrouterAnthropicRequestLayer = OpenrouterAnthropicRequestLayer;
const OPENROUTER_RESPONSES_LAYER: OpenrouterResponsesRequestLayer = OpenrouterResponsesRequestLayer;
const OPENROUTER_CHAT_COMPLETIONS_LAYER: OpenrouterChatCompletionsRequestLayer = OpenrouterChatCompletionsRequestLayer;

#[derive(Debug, Clone, Copy)]
struct OpenaiImagesRequestLayer;

impl RequestLayerImpl for OpenaiImagesRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/images/generations")
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }
}

impl OpenaiImagesRequestLayer {
  fn build_image_url(&self, base_url: &str, edit: bool) -> String {
    join_url(base_url, if edit { "/images/edits" } else { "/images/generations" })
  }
}

#[derive(Debug, Clone, Copy)]
struct FalRequestLayer;

impl RequestLayerImpl for FalRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, _stream: bool) -> String {
    join_url(base_url, &format!("/fal-ai/{model}"))
  }

  fn build_headers(&self, config: &BackendConfig, _stream: bool) -> Vec<(String, String)> {
    let mut headers = vec![(
      "Authorization".to_string(),
      format!("key {}", config.auth_token.expose()),
    )];
    headers.extend(config.headers.iter().map(|(key, value)| (key.clone(), value.clone())));
    headers
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentReferenceMode {
  Remote,
  Inline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentReferenceReason {
  NonUrlSource,
  UnsupportedScheme,
  GenericRemoteReference,
  GeminiApiFileUri,
  GeminiApiYoutubeUrl,
  GeminiApiInlineHttpUrl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentReferencePlan {
  pub mode: AttachmentReferenceMode,
  pub reason: AttachmentReferenceReason,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RequestIntentReasoning {
  #[serde(default)]
  pub enabled: bool,
  #[serde(default)]
  pub effort: Option<String>,
  #[serde(default)]
  pub budget_tokens: Option<u32>,
  #[serde(default)]
  pub include_reasoning: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RequestIntent {
  #[serde(default)]
  pub include: Vec<String>,
  #[serde(default)]
  pub reasoning: Option<RequestIntentReasoning>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResolvedRequestIntent {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub include: Option<Vec<String>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub reasoning: Option<Value>,
}

impl BackendRequestLayer {
  fn implementation(&self) -> &'static dyn RequestLayerImpl {
    match self {
      BackendRequestLayer::Anthropic => &ANTHROPIC_LAYER,
      BackendRequestLayer::ChatCompletions => &CHAT_COMPLETIONS_LAYER,
      BackendRequestLayer::CloudflareWorkersAi => &CLOUDFLARE_WORKERS_AI_LAYER,
      BackendRequestLayer::GeminiApi => &GEMINI_API_LAYER,
      BackendRequestLayer::GeminiVertex => &GEMINI_VERTEX_LAYER,
      BackendRequestLayer::PerplexitySonar => &PERPLEXITY_SONAR_LAYER,
      BackendRequestLayer::Responses => &RESPONSES_LAYER,
      BackendRequestLayer::VertexAnthropic => &VERTEX_ANTHROPIC_LAYER,
      BackendRequestLayer::OpenaiImages => &OPENAI_IMAGES_LAYER,
      BackendRequestLayer::Fal => &FAL_LAYER,
      BackendRequestLayer::OpenaiEmbeddings => &OPENAI_EMBEDDINGS_LAYER,
      BackendRequestLayer::OpenrouterAnthropic => &OPENROUTER_ANTHROPIC_LAYER,
      BackendRequestLayer::OpenrouterResponses => &OPENROUTER_RESPONSES_LAYER,
      BackendRequestLayer::OpenrouterChatCompletions => &OPENROUTER_CHAT_COMPLETIONS_LAYER,
    }
  }

  pub(super) fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    self.implementation().build_url(base_url, model, stream)
  }

  pub(super) fn build_image_url(&self, base_url: &str, model: &str, edit: bool) -> String {
    match self {
      BackendRequestLayer::OpenaiImages => OPENAI_IMAGES_LAYER.build_image_url(base_url, edit),
      _ => self.build_url(base_url, model, false),
    }
  }

  pub(super) fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    self.implementation().build_headers(config, stream)
  }

  pub(super) fn rewrite_body(&self, body: Value) -> Value {
    self.implementation().rewrite_body(body)
  }

  pub(super) fn build_embedding_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    self.implementation().build_embedding_url(base_url, model)
  }

  pub(super) fn build_embedding_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.implementation().build_embedding_headers(config)
  }

  pub(super) fn rewrite_embedding_body(&self, body: Value) -> Value {
    self.implementation().rewrite_embedding_body(body)
  }

  pub(super) fn build_rerank_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    self.implementation().build_rerank_url(base_url, model)
  }

  pub(super) fn build_rerank_headers(&self, config: &BackendConfig) -> Vec<(String, String)> {
    self.implementation().build_rerank_headers(config)
  }

  pub(super) fn rewrite_rerank_body(&self, body: Value) -> Value {
    self.implementation().rewrite_rerank_body(body)
  }

  pub fn plan_attachment_reference(&self, base_url: &str, source: &Value) -> AttachmentReferencePlan {
    let url = match source {
      Value::String(url) => Some(url.as_str()),
      Value::Object(object) => object.get("url").and_then(Value::as_str),
      _ => None,
    };

    let Some(url) = url else {
      return AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Inline,
        reason: AttachmentReferenceReason::NonUrlSource,
      };
    };

    match self {
      BackendRequestLayer::GeminiApi => plan_gemini_api_attachment_reference(url, base_url),
      _ => generic_remote_reference_plan(url),
    }
  }

  pub fn resolve_request_intent(&self, intent: RequestIntent) -> ResolvedRequestIntent {
    let mut include = intent.include;
    let mut reasoning = None;

    if let Some(reasoning_intent) = intent.reasoning.filter(|value| value.enabled) {
      if reasoning_intent.include_reasoning {
        include.push("reasoning".to_string());
      }

      reasoning = match self {
        BackendRequestLayer::Anthropic
        | BackendRequestLayer::VertexAnthropic
        | BackendRequestLayer::OpenrouterAnthropic => Some(Value::Object(
          [(
            "budget_tokens".to_string(),
            Value::from(reasoning_intent.budget_tokens.unwrap_or(12_000)),
          )]
          .into_iter()
          .collect(),
        )),
        _ => Some(Value::Object(match reasoning_intent.budget_tokens {
          Some(budget_tokens) => [("budget_tokens".to_string(), Value::from(budget_tokens))]
            .into_iter()
            .collect(),
          None => [(
            "effort".to_string(),
            Value::String(reasoning_intent.effort.unwrap_or_else(|| "medium".to_string())),
          )]
          .into_iter()
          .collect(),
        })),
      };
    }

    if !include.is_empty() {
      include.sort();
      include.dedup();
    }

    ResolvedRequestIntent {
      include: (!include.is_empty()).then_some(include),
      reasoning,
    }
  }

  pub(super) fn dispatch_rerank(
    &self,
    client: &dyn BackendHttpClient,
    config: &BackendConfig,
    protocol: &RerankProtocol,
    request: &RerankRequest,
  ) -> Result<Option<RerankResponse>, BackendError> {
    self.implementation().dispatch_rerank(client, config, protocol, request)
  }
}

pub(super) fn resolve_chat_request_layer(
  config: &BackendConfig,
  protocol: &ChatProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_chat_protocol(protocol));
  request_layer.ensure_chat_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_structured_request_layer(
  config: &BackendConfig,
  protocol: &StructuredProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_structured_protocol(protocol));
  request_layer.ensure_structured_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_embedding_request_layer(
  config: &BackendConfig,
  protocol: &EmbeddingProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_embedding_protocol(protocol));
  request_layer.ensure_embedding_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_rerank_request_layer(
  config: &BackendConfig,
  protocol: &RerankProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_rerank_protocol(protocol));
  request_layer.ensure_rerank_compatible(protocol)?;
  Ok(request_layer)
}

pub(super) fn resolve_image_request_layer(
  config: &BackendConfig,
  protocol: &ImageProtocol,
) -> Result<BackendRequestLayer, BackendError> {
  let request_layer = config
    .request_layer
    .unwrap_or_else(|| BackendRequestLayer::from_image_protocol(protocol));
  request_layer.ensure_image_compatible(protocol)?;
  Ok(request_layer)
}

pub fn resolve_attachment_reference_plan(
  config: &BackendConfig,
  protocol: &ChatProtocol,
  source: &Value,
) -> Result<AttachmentReferencePlan, BackendError> {
  let request_layer = resolve_chat_request_layer(config, protocol)?;
  Ok(request_layer.plan_attachment_reference(&config.base_url, source))
}

pub fn resolve_request_intent(
  config: &BackendConfig,
  protocol: &ChatProtocol,
  intent: RequestIntent,
) -> Result<ResolvedRequestIntent, BackendError> {
  let request_layer = resolve_chat_request_layer(config, protocol)?;
  Ok(request_layer.resolve_request_intent(intent))
}

pub(super) fn build_extra_headers(config: &BackendConfig) -> Vec<(String, String)> {
  let mut headers: Vec<(String, String)> = config
    .headers
    .iter()
    .map(|(key, value)| (key.clone(), value.clone()))
    .collect();
  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}

fn join_url(base_url: &str, path: &str) -> String {
  format!("{}{}", base_url.trim_end_matches('/'), path)
}

fn build_bearer_headers(config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
  let mut headers = vec![
    ("content-type".to_string(), "application/json".to_string()),
    (
      "accept".to_string(),
      if stream {
        "text/event-stream".to_string()
      } else {
        "application/json".to_string()
      },
    ),
  ];

  if !config.auth_token.is_empty() {
    headers.push((
      "authorization".to_string(),
      format!("Bearer {}", config.auth_token.expose()),
    ));
  }

  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}

fn build_api_key_headers(config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
  let mut headers = vec![
    ("content-type".to_string(), "application/json".to_string()),
    (
      "accept".to_string(),
      if stream {
        "text/event-stream".to_string()
      } else {
        "application/json".to_string()
      },
    ),
  ];

  if !config.auth_token.is_empty() {
    headers.push(("x-goog-api-key".to_string(), config.auth_token.expose().to_string()));
  }

  headers.sort_by(|a, b| a.0.cmp(&b.0));
  headers
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::{
    AttachmentReferenceMode, AttachmentReferencePlan, AttachmentReferenceReason, BackendRequestLayer, RequestIntent,
    RequestIntentReasoning,
  };

  #[test]
  fn gemini_api_should_inline_generic_http_attachments() {
    let plan = BackendRequestLayer::GeminiApi.plan_attachment_reference(
      "https://generativelanguage.googleapis.com/v1beta",
      &json!({ "url": "https://example.com/a.jpg" }),
    );

    assert_eq!(
      plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Inline,
        reason: AttachmentReferenceReason::GeminiApiInlineHttpUrl,
      }
    );
  }

  #[test]
  fn gemini_api_should_preserve_supported_remote_refs() {
    let base_url = "https://generativelanguage.googleapis.com/v1beta";

    let file_plan = BackendRequestLayer::GeminiApi.plan_attachment_reference(
      base_url,
      &json!({ "url": "https://generativelanguage.googleapis.com/v1beta/files/file-123" }),
    );
    assert_eq!(
      file_plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Remote,
        reason: AttachmentReferenceReason::GeminiApiFileUri,
      }
    );

    let youtube_plan = BackendRequestLayer::GeminiApi
      .plan_attachment_reference(base_url, &json!({ "url": "https://www.youtube.com/watch?v=abc123" }));
    assert_eq!(
      youtube_plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Remote,
        reason: AttachmentReferenceReason::GeminiApiYoutubeUrl,
      }
    );
  }

  #[test]
  fn gemini_vertex_should_preserve_remote_http_attachments() {
    let plan = BackendRequestLayer::GeminiVertex
      .plan_attachment_reference("https://vertex.example", &json!({ "url": "https://example.com/a.jpg" }));

    assert_eq!(
      plan,
      AttachmentReferencePlan {
        mode: AttachmentReferenceMode::Remote,
        reason: AttachmentReferenceReason::GenericRemoteReference,
      }
    );
  }

  #[test]
  fn anthropic_request_intent_should_force_budget_tokens() {
    let resolved = BackendRequestLayer::Anthropic.resolve_request_intent(RequestIntent {
      include: vec!["citations".to_string()],
      reasoning: Some(RequestIntentReasoning {
        enabled: true,
        effort: Some("high".to_string()),
        budget_tokens: None,
        include_reasoning: false,
      }),
    });

    assert_eq!(resolved.include, Some(vec!["citations".to_string()]));
    assert_eq!(resolved.reasoning, Some(json!({ "budget_tokens": 12000 })));
  }

  #[test]
  fn openai_request_intent_should_merge_reasoning_include_and_effort() {
    let resolved = BackendRequestLayer::ChatCompletions.resolve_request_intent(RequestIntent {
      include: vec!["citations".to_string()],
      reasoning: Some(RequestIntentReasoning {
        enabled: true,
        effort: Some("high".to_string()),
        budget_tokens: None,
        include_reasoning: true,
      }),
    });

    assert_eq!(
      resolved.include,
      Some(vec!["citations".to_string(), "reasoning".to_string()])
    );
    assert_eq!(resolved.reasoning, Some(json!({ "effort": "high" })));
  }
}
